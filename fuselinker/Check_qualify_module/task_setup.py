"""
Cho phép chạy main.py / main2.py ngay trong thư mục này.

Hai file đó là bản GỐC trước khi tách repo, nên vẫn `import myutils`,
`from calc_auroc import ...`, `from data_loader import Data` — những module giờ
nằm ở ../common/. File này đưa đúng các thư mục cần thiết lên sys.path:

    1. chính thư mục này  -> model.py, model_base2.py, model_base4.py bản gốc
    2. ../common          -> myutils, calc_auroc, data_loader, paths

Thứ tự đó quan trọng: model.py bản gốc ở đây phải được ưu tiên hơn bất cứ
model.py nào khác trong repo, để thư mục này là một ảnh chụp đông cứng dùng
cho việc đối chứng.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                  # fuselinker/
COMMON_DIR = ROOT / "common"

for _p in (HERE, COMMON_DIR):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)
