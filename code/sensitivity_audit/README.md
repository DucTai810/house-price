# Parameter sensitivity audit va tai lap

Thu muc nay dong goi phan sensitivity/reproducibility cua audit XAI:

1. Doi tham so audit: so diem giai thich, ty le che feature, kich thuoc nen/background.
2. Bao cao do nhay bang bootstrap 95% confidence interval.
3. Noi ket qua voi y tuong nhan qua/counterfactual cua giai thich.
4. Xuat bang, hinh va slide moi de nguoi khac kiem chung lai.

## Du lieu dau vao

Script khong train lai model. No dung cac artifact da co san tu King County:

- `code/king_county/outputs/xai_explain.parquet`
- `code/king_county/outputs/xai_bg.parquet`
- `code/king_county/outputs/artifacts.pkl`
- `code/king_county/outputs/model_*.pkl`
- `code/king_county/outputs/attr_*.npz`
- `code/king_county/outputs/consensus_attr.npz`

## Cai thu vien

Neu may dang co Python tot tai:

```powershell
C:\Users\ADMIN\AppData\Local\Python\bin\python.exe -m pip install -r code\sensitivity_audit\requirements.txt
```

## Chay lai sensitivity audit

Tu thu muc goc `House_Price`, chay:

```powershell
C:\Users\ADMIN\AppData\Local\Python\bin\python.exe code\sensitivity_audit\parameter_sensitivity.py
```

Neu can chay nhanh de test pipeline:

```powershell
C:\Users\ADMIN\AppData\Local\Python\bin\python.exe code\sensitivity_audit\parameter_sensitivity.py --bootstrap 150 --n-perturbations 12
```

## Dau ra

Script ghi vao `code/sensitivity_audit/outputs`:

- `sensitivity_ci.csv`: rank-correlation marginal vs on-manifold, kem 95% CI.
- `sensitivity_summary.csv`: do dao dong theo tung nhom tham so.
- `method_metric_means.csv`: metric trung binh theo tung method.
- `parameter_sensitivity.png`: hinh tong hop de chen vao slide.
- `parameter_sensitivity_report.md`: tom tat ket qua va dien giai nhan qua.

## Y nghia nhan qua

Marginal replacement tra loi cau hoi yeu hon: neu cat roi cac feature va thay bang gia tri trung binh, ranking XAI co con on khong? Cach nay de tao diem ngoai phan phoi du lieu.

On-manifold donor replacement gan voi cau hoi counterfactual hon: neu giu lai mot phan dac trung cua can nha va thay phan con lai bang mot can nha that gan nhat, giai thich co con dung khong? Cach nay khong chung minh nhan qua, nhung kiem tra xem giai thich co song sot duoc duoi can thiep hop ly hon khong.
