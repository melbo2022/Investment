# 金融電卓（Flask）
将来価値 / 現在価値 / 必要金利 / 必要年数 の4ページを備えたシンプルなWebアプリです。

## セットアップ
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

## ページ
- トップ：目的を選択
- 将来価値（/fv）
- 現在価値（/pv）
- 金利（/rate）
- 運用年数（/years）
