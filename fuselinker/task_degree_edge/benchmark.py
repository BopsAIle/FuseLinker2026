"""Sequential, matched-protocol SSL experiment; validation only unless --test.

Run from any directory. Model paths allow an immutable pre-edit snapshot.
Both encoders receive identical batches, degree supervision and eval negatives.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import time

import task_setup  # noqa: F401
import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import myutils
from lukepi_graph import generate_lukepi_masked_graph_and_labels
from ssl_eval import (UncertaintyWeighter, build_degree_labels,
                      encode_full_train_graph, evaluate_ssl_degree, evaluate_ssl_edge)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def load_data(folder):
    frames = [pd.read_csv(folder / (s + '.tsv'), sep='\t', header=None)
              for s in ('train', 'valid', 'test')]
    entities, relations = {}, {}
    arrays = []
    # Exactly Data's first-appearance order (head then tail per triple).
    for frame in frames:
        triples = []
        for src, rel, dst in frame.itertuples(index=False, name=None):
            s = entities.setdefault(src, len(entities))
            d = entities.setdefault(dst, len(entities))
            r = relations.setdefault(rel, len(relations))
            triples.append((s, r, d))
        arrays.append(np.asarray(triples, dtype=np.int64))
    return len(entities), len(relations), arrays


def main(args):
    torch.set_num_threads(4)
    seed_all(args.seed)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[1]
    folder = root / args.data
    model_path = Path(args.model).resolve()
    spec = importlib.util.spec_from_file_location('experiment_model', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    n, r, (train, valid, test) = load_data(folder)
    total = np.concatenate((train, valid, test))
    text = np.load(folder / 'pubmedbert_pretrained_embeddings_768.npy')
    domain = np.load(folder / 'poincare_embeddings.npy')
    if text.shape[0] != n or domain.shape[0] != n:
        raise ValueError('Embedding rows do not match entity IDs')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    extra = {} if model_path.name == 'model.py' else {'use_ppr': args.ppr}
    if getattr(module, 'MASK_SAFE_PPR', False):
        extra['ppr_c'] = args.ppr_c
    encoder = module.LinkPredict(
        n, 200, r, num_bases=20, num_hidden_layers=2, dropout=0.2, use_cuda=True,
        pretrained_text_embeddings=text, pretrained_domain_embeddings=domain,
        w=0.75, **extra).to(device)
    # model.py historically passes use_cuda positionally as use_self_loop.
    # Match the supplied training command rather than disabling its self loops.
    assert all(layer.self_loop for layer in encoder.rgcn.layers[1:])
    edge = module.EdgeTypeClassifier(200, r + 1, dropout=0.2).to(device)
    degree = module.DegreeClassifier(200, 10, dropout=0.2).to(device)
    weighting = UncertaintyWeighter().to(device)
    params = [p for m in (encoder, edge, degree, weighting) for p in m.parameters()]
    optimizer = torch.optim.Adam(params, lr=0.001)
    graph, rel, norm = myutils.build_graph(n, r, train)
    deg = graph.in_degrees().float().view(-1, 1)
    labels, _ = build_degree_labels(deg, 10, strategy='quantile')
    labels = labels.to(device)
    weights = labels.bincount(minlength=10).float().clamp_min(1).rsqrt()
    weights = weights * (10 / weights.sum())
    if hasattr(degree, 'configure_sampling_prior'):
        degree.configure_sampling_prior(deg, labels, weights, 1024, len(train))
    adj = myutils.get_adj(n, train)
    kwargs = {}
    if args.ppr and getattr(module, 'GLOBAL_MASKED_PPR', False):
        kwargs = dict(context_graph=graph.to(device))
    elif args.ppr and not getattr(module, 'MASK_SAFE_PPR', False):
        gp, wp = module.load_or_compute_ppr_sparse(
            n, train, cache_dir=str(args.output.parent / 'ppr_cache'),
            use_iterative=True, device=device)
        kwargs = dict(ppr_graph=gp, ppr_edge_weight=wp,
                      all_node_ids=torch.arange(n, device=device).view(-1, 1))

    def evaluate(split):
        encoder.eval(); edge.eval(); degree.eval()
        with torch.no_grad():
            h = encode_full_train_graph(encoder, graph, rel, norm, n, device, **kwargs)
            e = evaluate_ssl_edge(h, edge, split, r, 1, total, device,
                                  max_triples=5000, rng=np.random.RandomState(4242))
            d = evaluate_ssl_degree(h, degree, labels)
        return dict(edge=e, degree=d)

    report = dict(config=vars(args).copy(), model_sha256=hashlib.sha256(
        model_path.read_bytes()).hexdigest(), num_nodes=n, num_relations=r,
        protocol='subgraph degree; fixed validation; best validation edge CE',
        degree_prior_correction=hasattr(degree, 'configure_sampling_prior'), history=[])
    report['encoder_context'] = 'masked_global_ppr' if 'context_graph' in kwargs else 'batch_graph'
    report['config'] = {k: str(v) if isinstance(v, Path) else v
                        for k, v in report['config'].items()}
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location='cpu')
        for name, m in [('encoder', encoder), ('edge', edge), ('degree', degree)]:
            m.load_state_dict(saved[name])
        report['validation'] = evaluate(valid)
        if args.test:
            report['test'] = evaluate(test)
        report['seconds'] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
        return
    best_loss, best = float('inf'), None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.iterations + 1):
        encoder.train(); edge.train(); degree.train()
        # Independent batch RNG: architecture/dropout/PPR cannot change examples.
        np.random.seed(args.seed + step)
        g, ids, er, nnorm, pairs, targets = generate_lukepi_masked_graph_and_labels(
            train, 1024, 0.2, r, adj, deg, 1, 'uniform')
        en = myutils.node_norm_2_edge_norm(g, torch.from_numpy(nnorm).view(-1, 1))
        ids = torch.from_numpy(ids).long().to(device)
        pairs = torch.from_numpy(pairs).long().to(device)
        targets = torch.from_numpy(targets).long().to(device)
        optimizer.zero_grad(set_to_none=True)
        step_kwargs = dict(kwargs)
        if 'context_graph' in step_kwargs:
            step_kwargs['masked_pairs'] = ids[pairs[targets < r][:, [0, 2]]]
        h = encoder(g.to(device), ids.view(-1, 1),
                    torch.from_numpy(er).to(device), en.to(device), **step_kwargs)
        le = F.cross_entropy(edge(h[pairs[:, 0]], h[pairs[:, 2]]), targets)
        ld = F.cross_entropy(degree(h), labels[ids], weight=weights)
        loss = weighting(le, ld)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % args.evaluate_every == 0 or step == args.iterations:
            metrics = evaluate(valid)
            report['history'].append(dict(step=step, train_edge_loss=le.item(),
                                          train_degree_loss=ld.item(), **metrics))
            print(json.dumps(dict(step=step, **metrics)), flush=True)
            if metrics['edge']['loss'] < best_loss:
                best_loss = metrics['edge']['loss']
                report['best_step'] = step
                report['validation'] = metrics
                best = {name: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                        for name, m in [('encoder', encoder), ('edge', edge), ('degree', degree)]}
    if args.test:
        for name, m in [('encoder', encoder), ('edge', edge), ('degree', degree)]:
            m.load_state_dict(best[name])
        report['test'] = evaluate(test)
    report['seconds'] = time.monotonic() - started
    report['peak_gpu_mb'] = torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0
    torch.save(best, args.output.with_suffix('.pth'))
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('SAVED', args.output, 'seconds', report['seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', default='suppkg')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--evaluate-every', type=int, default=1)
    parser.add_argument('--ppr', action='store_true')
    parser.add_argument('--ppr-c', type=float, default=0.85)
    parser.add_argument('--test', action='store_true', help='Only after locking the architecture')
    parser.add_argument('--checkpoint', type=Path, help='Evaluate an existing benchmark checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
