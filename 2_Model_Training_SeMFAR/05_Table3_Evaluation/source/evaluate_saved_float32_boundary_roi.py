"""Evaluate saved float32 predictions with the fixed boundary-inclusive protocol."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import evaluate_float32_eroded_roi_edsr as ev
import recalculate_table3_boundary_refined as roi

def image(path):return np.asarray(Image.open(path).convert("L"),dtype=np.float32)/255.0

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--run",type=Path,required=True);ap.add_argument("--paired",type=Path,required=True);ap.add_argument("--prediction-dir",type=Path);ap.add_argument("--output-dir",type=Path);a=ap.parse_args();output=a.output_dir or a.run;output.mkdir(parents=True,exist_ok=True);pred_root=a.prediction_dir or (a.run/"float32_test_predictions")
    ck=torch.load(a.run/"best.pt",map_location="cpu",weights_only=False);names=json.loads((a.run/"split.json").read_text(encoding="utf-8-sig"))["test"];rows=[]
    for name in names:
        gt=image(a.paired/f"{name}_HR.png");pred=np.load(pred_root/f"{name}_pred_float32.npy")
        box,hi=roi.setup(gt);pi=ev.independent_init(pred,box);gm=roi.refine(gt,box,hi);pm=roi.refine(pred,box,pi)
        sm=roi.m.ssim_map(pred,gt);mse=float(np.mean((pred[gm]-gt[gm])**2));dice,iou,bf1,assd,hd95=ev.shape_metrics(pm,gm,tol=1)
        rows.append({"file":name,"roi_psnr":-10*np.log10(max(mse,1e-12)),"roi_ssim":float(sm[gm].mean()),"roi_dice":dice,"roi_iou":iou,"boundary_f1_tol1px":bf1,"assd_px":assd,"hd95_px":hd95,"hr_roi_pixels":int(gm.sum()),"pred_roi_pixels":int(pm.sum())})
    summary={"model":a.run.name,"n":len(rows),"best_epoch":int(ck["epoch"]),"checkpoint":str(a.run/"best.pt"),"prediction":"direct float32 inference","roi_ssim_mask":"complete refined HR mask, no erosion","segmentation":"independent Gaussian+Otsu initialization and morphological Chan-Vese","boundary_f1_tolerance_px":1,"distance_unit":"pixels"}
    for k in ("roi_psnr","roi_ssim","roi_dice","roi_iou","boundary_f1_tol1px","assd_px","hd95_px"):
        x=np.asarray([r[k] for r in rows]);summary[k+"_mean"]=float(x.mean());summary[k+"_std"]=float(x.std(ddof=1))
    stem="float32_boundary_inclusive_roi"
    with (output/f"{stem}_per_image.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (output/f"{stem}_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
