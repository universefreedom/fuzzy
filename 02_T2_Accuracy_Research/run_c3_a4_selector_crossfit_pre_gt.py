#!/usr/bin/env python3
"""Create fold-isolated label packages, then freeze C3 A4 T1/T2/T3 predictions."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

R=Path(__file__).resolve().parents[1];O=R/'artifacts/real822v3_c3_ifam_a4_crossfit_top1_selector_v1'
ID=O/'00_AUTHORITY/CANDIDATE_IDENTITY_MAP.csv';FE=O/'01_FEATURES/CANDIDATE_FEATURES.csv'
DETAIL=R/'artifacts/real822v3_c3_ifam_ocr_aware_restoration_gate_v1/14_OOF820_POST_FREEZE_MATERIAL_GATE/A0_A4_OOF820_POST_GT_DETAIL.csv'
FEATURES=json.loads((O/'01_FEATURES/FEATURE_SCHEMA.json').read_text(encoding='utf-8'))['model_feature_columns'];SEED=20260910
def rd(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def js(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(t,p)
def wr(p,rows,fields):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial')
 with t.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 os.replace(t,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def tie(s):return hashlib.sha256(s.encode()).hexdigest().upper()

def package():
 ids=rd(ID); feats={r['candidate_row_id']:r for r in rd(FE)}
 if set(feats)!={r['candidate_row_id'] for r in ids}:raise RuntimeError('IDENTITY_FEATURE_JOIN_FAIL')
 truth={r['group_id']:r['gt'] for r in rd(DETAIL) if r['arm_id']=='A4'}
 if len(truth)!=820:raise RuntimeError('GT_AUTHORITY_820_FAIL')
 for fold in range(5):
  d=O/f'02_SELECTORS/fold{fold}'; train={r['group_id'] for r in ids if int(r['outer_fold'])!=fold};test={r['group_id'] for r in ids if int(r['outer_fold'])==fold}
  if len(train)!=656 or len(test)!=164 or train&test:raise RuntimeError('FOLD_SPLIT_FAIL')
  # This file is produced by a packaging-only process. The predictor never opens DETAIL.
  wr(d/'TRAIN_LABELS_ONLY.csv',({'group_id':g,'gt':truth[g]} for g in sorted(train)),['group_id','gt'])
  js(d/'PREDICTOR_INPUT_CONTRACT.json',{'outer_fold':fold,'train_group_ids':sorted(train),'test_group_ids':sorted(test),'train_labels_sha256':sha(d/'TRAIN_LABELS_ONLY.csv'),'test_GT_delivered':False,'test_GT_rows':0,'predictor_forbidden_input':str(DETAIL),'feature_sha256':sha(FE),'identity_sha256':sha(ID)})

def predict(fold):
 d=O/f'02_SELECTORS/fold{fold}'; c=json.loads((d/'PREDICTOR_INPUT_CONTRACT.json').read_text(encoding='utf-8'));ids=rd(ID);f={r['candidate_row_id']:r for r in rd(FE)};lab=rd(d/'TRAIN_LABELS_ONLY.csv')
 train_ids=set(c['train_group_ids']);test_ids=set(c['test_group_ids']); labels={r['group_id']:r['gt'] for r in lab}
 if set(labels)!=train_ids or set(labels)&test_ids or len(lab)!=656:raise RuntimeError('TEST_GT_DELIVERY_FIREWALL')
 tr=[r for r in ids if r['group_id'] in train_ids];te=[r for r in ids if r['group_id'] in test_ids]
 X=np.asarray([[float(f[r['candidate_row_id']][k]) for k in FEATURES] for r in tr]);y=np.asarray([int(r['candidate_string']==labels[r['group_id']]) for r in tr])
 sizes={g:sum(r['group_id']==g for r in tr) for g in train_ids};w=np.asarray([1/sizes[r['group_id']] for r in tr])
 if set(y)!={0,1}:raise RuntimeError('TARGET_DEGENERATE')
 sc=StandardScaler().fit(X);m=LogisticRegression(C=1.,penalty='l2',solver='liblinear',random_state=SEED,max_iter=1000).fit(sc.transform(X),y,sample_weight=w)
 by=defaultdict(list)
 for r in te:by[r['group_id']].append(r)
 t1=[];t2=[];t3=[]
 for g in sorted(by):
  rr=by[g];xx=np.asarray([[float(f[r['candidate_row_id']][k]) for k in FEATURES] for r in rr]);scores=m.predict_proba(sc.transform(xx))[:,1]
  for r,s in zip(rr,scores):r['_s']=float(s)
  # T1 follows the preregistered evidence ordering; lexical identity is never used.
  k1=lambda r:(int(f[r['candidate_row_id']]['is_c3_anchor']),int(f[r['candidate_row_id']]['view_source_count']),float(f[r['candidate_row_id']]['confidence_max']),tie(r['candidate_string']))
  a=max(rr,key=k1);b=max(rr,key=lambda r:(r['_s'],tie(r['candidate_string'])));base=next(r for r in rr if int(f[r['candidate_row_id']]['is_c3_anchor']))
  fb=f[b['candidate_row_id']]; valid=len(b['candidate_string']) in (7,8)
  safe=b if b['_s']>base['_s'] and valid and int(fb['view_source_count'])>=2 else base
  for dest,row in ((t1,a),(t2,b),(t3,safe)):dest.append({'group_id':g,'prediction':row['candidate_string'],'candidate_row_id':row['candidate_row_id']})
 model=d/'SELECTOR_MODEL_PRE_GT.json';js(model,{'outer_fold':fold,'features':FEATURES,'string_identity_feature_count':0,'C':1.,'seed':SEED,'group_weight':'1/|C_g|','group_weight_total_min':min(sum(w[i] for i,r in enumerate(tr) if r['group_id']==g) for g in train_ids),'group_weight_total_max':max(sum(w[i] for i,r in enumerate(tr) if r['group_id']==g) for g in train_ids),'coefficient':m.coef_[0].tolist(),'intercept':float(m.intercept_[0]),'scaler_mean':sc.mean_.tolist(),'scaler_scale':sc.scale_.tolist(),'test_GT_reads':0})
 for name,rows in (('T1',t1),('T2',t2),('T3',t3)):js(d/f'{name}_PRE_GT.json',{'outer_fold':fold,'groups':164,'predictions':rows,'test_GT_reads':0})
 universe={(r['group_id'],r['candidate_row_id']) for r in te};new=sum((r['group_id'],r['candidate_row_id']) not in universe for rows in (t1,t2,t3) for r in rows)
 js(d/'PREDICTION_FREEZE.json',{'status':'PRE_GT_FREEZE_PASS','outer_fold':fold,'groups':164,'test_GT_delivered':False,'test_GT_reads':0,'new_strings':new,'model_sha256':sha(model),**{f'{n}_sha256':sha(d/f'{n}_PRE_GT.json') for n in ('T1','T2','T3')}})

def merge():
 out={n:[] for n in ('T1','T2','T3')};folds=[]
 for k in range(5):
  fr=json.loads((O/f'02_SELECTORS/fold{k}/PREDICTION_FREEZE.json').read_text(encoding='utf-8'));folds.append(fr)
  for n in out:out[n]+=json.loads((O/f'02_SELECTORS/fold{k}/{n}_PRE_GT.json').read_text(encoding='utf-8'))['predictions']
 for n,rows in out.items():
  if len(rows)!=820 or len({r['group_id'] for r in rows})!=820:raise RuntimeError('MERGE_IDENTITY_FAIL')
  js(O/f'03_PRE_GT/{n}_OOF820_PRE_GT.json',{'groups':820,'predictions':sorted(rows,key=lambda r:r['group_id']),'GT_reads':0})
 js(O/'03_PRE_GT/PRE_GT_SHA_AUDIT.json',{'status':'PRE_GT_OOF820_FREEZE_PASS','folds':folds,'missing':0,'duplicates':0,'new_strings':0,'test_GT_reads':0,**{f'{n}_sha256':sha(O/f'03_PRE_GT/{n}_OOF820_PRE_GT.json') for n in out}})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('package','predict','merge'));p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args()
 if a.mode=='package':package()
 elif a.mode=='predict':
  if a.fold is None:raise SystemExit('--fold required')
  predict(a.fold)
 else:merge()
