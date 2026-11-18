"""Mesin eksperimen final tiga backbone, S2/S3/S4, dan holdout terpusat."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import random
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm

from ecgtl.hierarchical.data import read_manifest
from ecgtl.hierarchical.metrics import calculate, export_metrics
from ecgtl.hierarchical.model import backbone, loss_terms
from ecgtl.hierarchical.protocol import atomic_json, atomic_torch, digest, environment, seed_all
from ecgtl.hierarchical.training import Experiment, load_checkpoint, to_device
from tools.a5_physiology import (
    GLOBAL_PHYSIOLOGY_NAMES, LOCAL_PHYSIOLOGY_NAMES, _append_global_morphology,
    _beat_features, _global_rr_features,
)
from tools.a6_pb_nonlinear_fusion import NonlinearHead
from tools.a6_wave_delineation import (
    GLOBAL_WAVE_NAMES, LOCAL_WAVE_NAMES, extract_wave_features, summarize_wave_features,
)
from tools.rr_ablation_a4_sequence import A4SequenceFeatureStore, A4SequenceScaler
from tools.rr_ablation_a6_standalone import A6Dataset, collate_a6


FOLDER = Path(__file__).resolve().parent
DIMENSIONS = {'efficientnet_b0': 1280, 'mobilenet_v2': 1280, 'shufflenet_v2_x1_0': 1024}
SCENARIOS = ('S2', 'S3', 'S4')
TAU = {'S2': 1.0, 'S3': 2.0, 'S4': 2.0}
ENCODER_PROBABILITY_TOLERANCE = 1e-3
INITIALIZATION_ANCHOR_WINDOWS = 64


def _json_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def method_hash() -> str:
    code_root = FOLDER.parent
    files = [
        FOLDER/'final_v2.yaml', FOLDER/'METHOD_LOCK.json', Path(__file__),
        *(code_root/'src/ecgtl/hierarchical'/name for name in [
            'data.py', 'metrics.py', 'model.py', 'protocol.py', 'training.py',
        ]),
        *(code_root/'tools'/name for name in [
            'a5_physiology.py', 'a6_pb_nonlinear_fusion.py',
            'a6_wave_delineation.py', 'rr_ablation.py',
            'rr_ablation_a2.py', 'rr_ablation_a4_sequence.py',
            'rr_ablation_a5_physiology.py', 'rr_ablation_a5_standalone.py',
            'rr_ablation_a6_patient_balanced.py',
            'rr_ablation_a6_standalone.py',
        ]),
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Berkas implementasi metode tidak lengkap: {missing}')
    return hashlib.sha256(b''.join(
        str(path.relative_to(code_root)).encode() + path.read_bytes()
        for path in files
    )).hexdigest()


def initialize(protocol) -> dict:
    """Buat/verifikasi lock protokol data dan metode final pada output baru."""
    if not (protocol.out/'protocol_lock.json').exists():
        protocol.lock()
    else:
        protocol.verify()
    payload = {'method_hash': method_hash(), 'data_protocol_hash': protocol.hash,
               'method_lock_sha256': digest(FOLDER/'METHOD_LOCK.json'),
               'holdout_used_for_selection': False}
    path = protocol.out/'final_method_lock.json'
    if path.exists() and json.loads(path.read_text()) != payload:
        raise ValueError('Metode final berubah setelah output dibuat; gunakan output baru.')
    atomic_json(path, payload)
    return payload


def _feature_source_hash(protocol, windows, local) -> str:
    implementation = hashlib.sha256(
        inspect.getsource(build_label_free_feature_cache).encode() +
        (FOLDER.parent/'tools/a5_physiology.py').read_bytes() +
        (FOLDER.parent/'tools/a6_wave_delineation.py').read_bytes()
    ).hexdigest()
    return _json_hash({
        'data_protocol_hash': protocol.hash,
        'feature_implementation_hash': implementation,
        'window_ids': hashlib.sha256('\n'.join(windows.window_id.astype(str)).encode()).hexdigest(),
        'crop_ids': hashlib.sha256('\n'.join(local.crop_id.astype(str)).encode()).hexdigest(),
        'physiology_names': [GLOBAL_PHYSIOLOGY_NAMES, LOCAL_PHYSIOLOGY_NAMES],
        'wave_names': [GLOBAL_WAVE_NAMES, LOCAL_WAVE_NAMES],
    })


def build_label_free_feature_cache(protocol) -> Path:
    """Ekstrak fisiologi/delineasi semua window tanpa membaca kolom label."""
    initialize(protocol)
    windows = read_manifest(protocol.prepared/'global_manifest.csv').copy()
    local = read_manifest(protocol.prepared/'local_manifest.csv').copy()
    windows['window_id'] = windows.window_id.astype(str)
    local['parent_window_id'] = local.parent_window_id.astype(str)
    local = local[local.parent_window_id.isin(set(windows.window_id))]
    source_hash = _feature_source_hash(protocol, windows, local)
    folder = protocol.out/'feature_store'/source_hash[:16]
    marker = folder/'complete.json'
    if marker.exists():
        meta = json.loads(marker.read_text())
        if meta['source_hash'] != source_hash or digest(folder/'features.npz') != meta['features_sha256']:
            raise ValueError('Integritas feature store final tidak sesuai.')
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    groups = {key: frame.sort_values('rpeak_sample').reset_index(drop=True)
              for key, frame in local.groupby('parent_window_id', sort=False)}
    ids, pg, pl, wg, wl, offsets = [], [], [], [], [], [0]
    fs = float(protocol.cfg['data']['fs'])
    # Label dan split tidak dipakai dalam ekstraksi. Urutan mengikuti global manifest canonical.
    for (database, record), rows in tqdm(windows.groupby(['database','record'], sort=False),
                                         desc='Fitur fisiologi/delineasi final'):
        record_folder = protocol.prepared/'records'/str(database)/str(record)
        waveforms = np.load(record_folder/'waveforms.npy', mmap_mode='r')
        indices = np.load(record_folder/'window_indices.npy')
        positions = {int(value): index for index, value in enumerate(indices)}
        for row in rows.itertuples(index=False):
            wid = str(row.window_id)
            crop_rows = groups[wid]
            waveform = np.asarray(waveforms[positions[int(row.index)]], dtype=np.float64)
            peaks = crop_rows.rpeak_sample.to_numpy(np.int64) - int(row.start_target_fs)
            if np.any(peaks < 0) or np.any(peaks >= len(waveform)):
                raise ValueError(f'R-peak di luar window: {wid}')
            beat = _beat_features(waveform, peaks, fs)
            phys_global = _append_global_morphology(_global_rr_features(peaks, fs), beat)
            wave_global, wave_local = extract_wave_features(waveform, peaks, fs)
            fallback = crop_rows.fallback_peak.astype(bool).to_numpy()
            if fallback.any():
                # Metode pilot terkunci mempertahankan fitur fisiologi/morfologi
                # pada fallback crop. Hanya delineasi P/PR/QRS yang menandai
                # fallback sebagai missing, sesuai build_wave_cache A6.
                wave_local[fallback] = np.nan
                wave_global = summarize_wave_features(wave_local)
            ids.append(wid); pg.append(phys_global); pl.append(beat)
            wg.append(wave_global); wl.append(wave_local); offsets.append(offsets[-1] + len(beat))
    if ids != windows.window_id.astype(str).tolist():
        raise ValueError('Urutan feature store tidak sama dengan manifest canonical.')
    with (folder/'features.tmp').open('wb') as stream:
        np.savez_compressed(stream, window_ids=np.asarray(ids),
            physiology_global=np.stack(pg).astype(np.float32),
            physiology_local=np.concatenate(pl).astype(np.float32),
            wave_global=np.stack(wg).astype(np.float32),
            wave_local=np.concatenate(wl).astype(np.float32),
            local_offsets=np.asarray(offsets,dtype=np.int64))
    os.replace(folder/'features.tmp', folder/'features.npz')
    atomic_json(marker, {'source_hash':source_hash,'data_protocol_hash':protocol.hash,
        'windows':len(ids),'development_windows':int((windows.split=='development').sum()),
        'holdout_windows':int((windows.split=='holdout').sum()),'label_columns_read_for_features':False,
        'features_sha256':digest(folder/'features.npz')})
    return folder


def _compute_rr_sequence_arrays(windows, local, fs):
    """Hitung fitur RR yang identik dengan A2/A4 tanpa groupby DataFrame per window."""
    windows = windows.copy()
    local = local.copy()
    windows['window_id'] = windows.window_id.astype(str)
    local['parent_window_id'] = local.parent_window_id.astype(str)
    order = {wid:index for index,wid in enumerate(windows.window_id)}
    local['_window_order'] = local.parent_window_id.map(order)
    if local['_window_order'].isna().any():
        raise ValueError('Local manifest memuat window di luar global manifest.')
    local = local.sort_values(['_window_order','rpeak_sample'],kind='stable').reset_index(drop=True)
    grouped = local.groupby('parent_window_id',sort=False).size()
    if grouped.index.astype(str).tolist() != windows.window_id.tolist():
        raise ValueError('Urutan/local coverage tidak sama dengan global manifest.')
    offsets = np.r_[0,grouped.cumsum().to_numpy(np.int64)]
    global_rr = np.full((len(windows),10),np.nan,dtype=np.float32)
    local_rr = np.full((len(local),7),np.nan,dtype=np.float32)
    sequence = np.full((len(local),3),np.nan,dtype=np.float32)
    peaks_all = local.rpeak_sample.to_numpy(np.float64)
    fallback_all = local.fallback_peak.astype(bool).to_numpy()
    database = local.database.astype(str).to_numpy()
    record = local.record_id.astype(str).to_numpy()
    channel = local.channel.astype(str).to_numpy()
    record_parts = {}
    record_keys = []
    for index in range(len(windows)):
        start,stop=int(offsets[index]),int(offsets[index+1])
        peaks=peaks_all[start:stop]; fallback=fallback_all[start:stop]
        if len(peaks)>1 and np.any(np.diff(peaks)<0):
            raise ValueError(f'Urutan R-peak tidak monoton: {windows.window_id.iloc[index]}')
        valid_peaks=peaks[~fallback]
        rr=np.diff(valid_peaks)/fs if len(valid_peaks)>1 else np.empty(0)
        median=float(np.median(rr)) if len(rr) else np.nan
        global_rr[index,0]=len(valid_peaks)
        if len(rr):
            mean_rr=float(np.mean(rr)); q25,q75=np.percentile(rr,[25,75])
            global_rr[index,1]=median
            global_rr[index,2]=60.0/mean_rr if mean_rr>0 else np.nan
            global_rr[index,3]=float(np.std(rr,ddof=1)) if len(rr)>1 else np.nan
            global_rr[index,4]=float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr)>1 else np.nan
            global_rr[index,5]=float(np.std(rr)/mean_rr) if mean_rr>0 else np.nan
            global_rr[index,6]=float(q75-q25)
            global_rr[index,7]=float(np.min(rr)/median) if median>0 else np.nan
            global_rr[index,8]=float(np.max(rr)/median) if median>0 else np.nan
        if not fallback.any() and len(peaks):
            previous=np.r_[np.nan,np.diff(peaks)/fs]
            following=np.r_[np.diff(peaks)/fs,np.nan]
            local_rr[start:stop,0]=previous; local_rr[start:stop,1]=following
            with np.errstate(divide='ignore',invalid='ignore'):
                local_rr[start:stop,2]=previous/median
                local_rr[start:stop,3]=following/median
                local_rr[start:stop,4]=following/previous
                local_rr[start:stop,5]=(previous+following)/(2*median)
        key=f'{database[start]}/{record[start]}/{channel[start]}'
        record_keys.append(key)
        if not fallback.any():
            record_values=np.diff(peaks)/fs
            record_values=record_values[np.isfinite(record_values)&(record_values>0)]
            if len(record_values): record_parts.setdefault(key,[]).append(record_values)
    record_median={key:float(np.median(np.concatenate(parts))) for key,parts in record_parts.items()}
    for index,key in enumerate(record_keys):
        start,stop=int(offsets[index]),int(offsets[index+1])
        baseline=record_median.get(key,np.nan); global_rr[index,9]=baseline
        if np.isfinite(baseline) and baseline>0:
            previous=local_rr[start:stop,0]; following=local_rr[start:stop,1]
            local_rr[start:stop,6]=previous/baseline
            sequence[start:stop,0]=previous/baseline
            sequence[start:stop,1]=following/baseline
            sequence[start:stop,2]=(following-previous)/baseline
    return global_rr,local_rr,sequence,offsets


def build_rr_sequence_cache(protocol, feature_folder=None) -> Path:
    """Persistenkan RR/dRR A2/A4 agar inisialisasi eksperimen tidak mengulang 48k groupby."""
    initialize(protocol)
    folder=Path(feature_folder or build_label_free_feature_cache(protocol))
    target=folder/'rr_sequence.npz'; marker=folder/'rr_sequence_complete.json'
    windows=read_manifest(protocol.prepared/'global_manifest.csv').copy()
    local=read_manifest(protocol.prepared/'local_manifest.csv').copy()
    source_hash=_json_hash({'schema':'a2_a4_rr_sequence_v1','data_protocol_hash':protocol.hash,
        'implementation_hash':hashlib.sha256(
            inspect.getsource(_compute_rr_sequence_arrays).encode()).hexdigest(),
        'window_ids':hashlib.sha256('\n'.join(windows.window_id.astype(str)).encode()).hexdigest(),
        'crop_ids':hashlib.sha256('\n'.join(local.crop_id.astype(str)).encode()).hexdigest(),
        'fs':float(protocol.cfg['data']['fs'])})
    if marker.exists():
        receipt=json.loads(marker.read_text())
        if receipt.get('source_hash')!=source_hash or not target.is_file() or digest(target)!=receipt.get('sha256'):
            raise ValueError('Integritas cache RR/sequence final tidak sesuai.')
        return target
    global_rr,local_rr,sequence,offsets=_compute_rr_sequence_arrays(
        windows,local,float(protocol.cfg['data']['fs']))
    temporary=target.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,window_ids=np.asarray(windows.window_id.astype(str).tolist(),dtype=np.str_),
            global_rr=global_rr,local_rr=local_rr,sequence=sequence,local_offsets=offsets)
    os.replace(temporary,target)
    atomic_json(marker,{'source_hash':source_hash,'sha256':digest(target),
        'windows':len(windows),'local_beats':len(local),'method':'A2/A4 exact vectorized cache'})
    return target


class FinalFeatureStore(A4SequenceFeatureStore):
    def __init__(self, protocol, allowed_window_ids):
        allowed = list(map(str, allowed_window_ids))
        self.final_cache = build_label_free_feature_cache(protocol)
        rr_path = build_rr_sequence_cache(protocol,self.final_cache)
        values = np.load(self.final_cache/'features.npz')
        rr_values = np.load(rr_path)
        ids = values['window_ids'].astype(str)
        rr_ids = rr_values['window_ids'].astype(str)
        if not np.array_equal(ids,rr_ids):
            raise ValueError('Urutan cache fitur dan RR/sequence berbeda.')
        position = {wid:index for index,wid in enumerate(ids)}
        if set(allowed) - set(position):
            raise ValueError('Feature store final tidak mencakup window yang diminta.')
        offsets = values['local_offsets']
        rr_offsets = rr_values['local_offsets']
        if not np.array_equal(offsets,rr_offsets):
            raise ValueError('Offset lokal cache fitur dan RR/sequence berbeda.')
        # NpzFile mendekompresi ulang setiap kali __getitem__ dipanggil. Muat setiap
        # array tepat satu kali sebelum loop agar 24 ribu window tidak memicu
        # puluhan ribu dekompresi yang identik.
        physiology_global = values['physiology_global']
        physiology_local = values['physiology_local']
        wave_global = values['wave_global']
        wave_local = values['wave_local']
        rr_global = rr_values['global_rr']
        rr_local = rr_values['local_rr']
        rr_sequence = rr_values['sequence']
        self.by_window, self.sequence_by_window = {}, {}
        self.physiology_by_window, self.wave_by_window = {}, {}
        for wid in allowed:
            index = position[wid]; sl = slice(offsets[index], offsets[index+1])
            self.by_window[wid] = (rr_global[index].copy(),rr_local[sl].copy())
            self.sequence_by_window[wid] = rr_sequence[sl].copy()
            self.physiology_by_window[wid] = (physiology_global[index].copy(),
                                               physiology_local[sl].copy())
            self.wave_by_window[wid] = (wave_global[index].copy(),wave_local[sl].copy())


class FinalScaler(A4SequenceScaler):
    def __init__(self, store, window_ids):
        ids = list(map(str, window_ids))
        super().__init__(store, ids)
        for prefix, source in [('physiology',store.physiology_by_window),('wave',store.wave_by_window)]:
            global_values = np.stack([source[wid][0] for wid in ids]).astype(np.float64)
            local_values = np.concatenate([source[wid][1] for wid in ids]).astype(np.float64)
            global_mean, global_std = self._fit(global_values)
            local_mean, local_std = self._fit(local_values)
            setattr(self,f'{prefix}_global_mean',global_mean)
            setattr(self,f'{prefix}_global_std',global_std)
            setattr(self,f'{prefix}_local_mean',local_mean)
            setattr(self,f'{prefix}_local_std',local_std)

    def transform_extra(self, prefix, scope, values):
        return self._transform(values,getattr(self,f'{prefix}_{scope}_mean'),getattr(self,f'{prefix}_{scope}_std'))

    def state_dict(self):
        state = super().state_dict()
        for prefix in ['physiology','wave']:
            for scope in ['global','local']:
                state[f'{prefix}_{scope}_mean'] = getattr(self,f'{prefix}_{scope}_mean').tolist()
                state[f'{prefix}_{scope}_std'] = getattr(self,f'{prefix}_{scope}_std').tolist()
        return state


class FinalDataset(A6Dataset):
    def __getitem__(self,index):
        item = self.base[index]
        wid = str(item['id'])
        global_rr, local_rr = self.store.by_window[wid]
        sequence = self.store.sequence_by_window[wid]
        physiology_global, physiology_local = self.store.physiology_by_window[wid]
        wave_global, wave_local = self.store.wave_by_window[wid]
        count = len(item['local'])
        if any(len(values) != count for values in [local_rr,sequence,physiology_local,wave_local]):
            raise ValueError(f'Jumlah fitur lokal berbeda pada {wid}.')
        item['global_rr'] = torch.from_numpy(self.scaler.transform_global(global_rr))
        item['local_rr'] = torch.from_numpy(self.scaler.transform_local(local_rr))
        item['rr_sequence'] = torch.from_numpy(self.scaler.transform_sequence(sequence))
        item['physiology_global'] = torch.from_numpy(self.scaler.transform_extra('physiology','global',physiology_global))
        item['physiology_local'] = torch.from_numpy(self.scaler.transform_extra('physiology','local',physiology_local))
        item['wave_global'] = torch.from_numpy(self.scaler.transform_extra('wave','global',wave_global))
        item['wave_local'] = torch.from_numpy(self.scaler.transform_extra('wave','local',wave_local))
        return item


class EncoderHeadNet(nn.Module):
    def __init__(self, arch, cfg, head=None, trainable_groups=0):
        super().__init__()
        self.encoder, self.dim = backbone(arch, pretrained=True)
        self.head = head or NonlinearHead(self.dim,cfg['model']['dropout'])
        self.chunk = int(cfg['model']['feature_chunk'])
        self.trainable_groups = int(trainable_groups)
        for parameter in self.encoder.parameters(): parameter.requires_grad = False
        if hasattr(self.encoder,'features'):
            groups = list(self.encoder.features)
        else:
            groups = [self.encoder.conv1,self.encoder.stage2,self.encoder.stage3,
                      self.encoder.stage4,self.encoder.conv5]
        self.unfrozen_groups = groups[-self.trainable_groups:] if self.trainable_groups else []
        for group in self.unfrozen_groups:
            for parameter in group.parameters(): parameter.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if self.trainable_groups:
            self.encoder.eval()
            for group in self.unfrozen_groups: group.train(mode)
            for module in self.encoder.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm): module.eval()
        return self

    def encode(self, images):
        return torch.cat([self.encoder(part) for part in images.split(self.chunk)])

    def forward(self,global_x,local_x,mask,*features):
        global_features = self.encode(global_x)
        flat = self.encode(local_x[mask])
        local_features = flat.new_zeros((*mask.shape,self.dim)); local_features[mask] = flat
        return self.head(global_features,local_features,mask,*features)


class FinalExperiment:
    def __init__(self, protocol, arch, scenario, device=None, resume=True):
        if arch not in DIMENSIONS or scenario not in SCENARIOS: raise ValueError('Backbone/skenario tidak valid.')
        initialize(protocol)
        self.p,self.arch,self.scenario,self.resume = protocol,arch,scenario,bool(resume)
        self.base = Experiment(protocol,arch,device=device,resume=resume)
        self.device,self.cfg,self.dev = self.base.device,self.base.cfg,self.base.dev
        manifest = read_manifest(protocol.prepared/'global_manifest.csv')
        self.all_rows = manifest.reset_index(drop=True)
        self.store = FinalFeatureStore(protocol,self.all_rows.window_id)
        self.cache = self.base.cache

    def run_dir(self,fold,full=False):
        return self.p.out/('final_models' if full else 'cv')/self.arch/self.scenario/('final' if full else f'fold_{int(fold)}')

    def source_dir(self,fold,full=False):
        return self.p.out/('final_models' if full else 'cv')/self.arch/'S3'/('final' if full else f'fold_{int(fold)}')

    @staticmethod
    def sampling_weights(rows):
        counts=rows.patient_id.astype(str).value_counts()
        return rows.patient_id.astype(str).map((1/counts).to_dict()).to_numpy(np.float64)

    def loader(self,rows,scaler,cached,shuffle=False,seed=42):
        rows=rows.reset_index(drop=True); generator=torch.Generator().manual_seed(seed)
        dataset=FinalDataset(self.p,rows,self.cache if cached else None,self.store,scaler)
        sampler=None
        if shuffle:
            sampler=WeightedRandomSampler(torch.tensor(self.sampling_weights(rows),dtype=torch.double),
                num_samples=len(rows),replacement=True,generator=generator)
        return DataLoader(dataset,batch_size=self.cfg['batch_size'],shuffle=False,sampler=sampler,
            collate_fn=collate_a6,num_workers=self.cfg['num_workers'],generator=generator,
            pin_memory=self.device.type=='cuda')

    @staticmethod
    def forward(model,batch):
        return model(batch['global'],batch['local'],batch['mask'],batch['global_rr'],batch['local_rr'],
            batch['rr_sequence'],batch['physiology_global'],batch['physiology_local'],
            batch['wave_global'],batch['wave_local'])

    def _head(self): return NonlinearHead(DIMENSIONS[self.arch],self.p.cfg['model']['dropout']).to(self.device)

    def _fit_image_scaler(self,head,rows): self.base.fit_scaler(head,rows)

    def _model_scaler(self,train,fold,full,last_exists):
        scaler=FinalScaler(self.store,train.window_id)
        if self.scenario in ('S2','S3'):
            model=self._head(); self._fit_image_scaler(model,train); cached=True
        else:
            source=load_checkpoint(self.source_dir(fold,full)/'best.pt')
            if source['rr_scaler'] != scaler.state_dict(): raise ValueError('Scaler S3 sumber berbeda.')
            head=self._head(); head.load_state_dict(source['model'])
            model=EncoderHeadNet(self.arch,self.p.cfg,head,2).to(self.device); cached=False
        return model,scaler,cached

    def _optimizer(self,model):
        if self.scenario!='S4':
            return torch.optim.AdamW(model.parameters(),lr=self.cfg['lr_frozen'],weight_decay=self.cfg['weight_decay'])
        backbone=[p for p in model.encoder.parameters() if p.requires_grad]
        return torch.optim.AdamW([{'params':backbone,'lr':self.cfg['lr_finetune_backbone']},
            {'params':model.head.parameters(),'lr':self.cfg['lr_finetune_head']}],weight_decay=self.cfg['weight_decay'])

    def _initialization_tolerance(self):
        return ENCODER_PROBABILITY_TOLERANCE

    def _audit_raw_anchor(self,model,scaler,source_folder):
        anchor_path=Path(source_folder)/'initialization_anchor.npz'
        if not anchor_path.exists():
            raise FileNotFoundError(f'Anchor encoder tidak tersedia: {anchor_path}')
        anchor=np.load(anchor_path)
        ids=anchor['window_ids'].astype(str).tolist()
        indexed=self.dev.copy()
        indexed.index=indexed.window_id.astype(str)
        rows=indexed.loc[ids].reset_index(drop=True)
        prediction=self.predict(model,rows,scaler,False)
        expected_y=anchor['labels']; expected_probability=anchor['probabilities']
        maximum=float(np.max(np.abs(prediction['prob']-expected_probability)))
        audit={'windows':len(rows),
            'labels_identical':bool(np.array_equal(prediction['y'],expected_y)),
            'predicted_classes_identical':bool(np.array_equal(
                prediction['prob'].argmax(1),expected_probability.argmax(1))),
            'maximum_probability_difference':maximum,
            'tolerance':self._initialization_tolerance()}
        if (not audit['labels_identical'] or not audit['predicted_classes_identical'] or
                maximum>audit['tolerance']):
            raise ValueError(f'Raw encoder tidak mereproduksi anchor frozen: {audit}')
        return audit,prediction

    def _audit_s4_initialization(self,model,scaler,fold,full,valid):
        """Pastikan raw encoder S4 mereproduksi prediksi S3 sebelum fine-tuning."""
        source_folder=self.source_dir(fold,full)
        if full:
            audit,_=self._audit_raw_anchor(model,scaler,source_folder)
            return {'fold':int(fold),'full':True,**audit},None
        else:
            rows=valid
            expected_y=np.load(source_folder/'labels.npy')
            expected_probability=np.load(source_folder/'probabilities.npy')
        prediction=self.predict(model,rows,scaler,False)
        maximum=float(np.max(np.abs(prediction['prob']-expected_probability)))
        audit={'fold':int(fold),'full':bool(full),'windows':len(rows),
            'labels_identical':bool(np.array_equal(prediction['y'],expected_y)),
            'predicted_classes_identical':bool(np.array_equal(
                prediction['prob'].argmax(1),expected_probability.argmax(1))),
            'maximum_probability_difference':maximum,
            'tolerance':self._initialization_tolerance()}
        if (not audit['labels_identical'] or not audit['predicted_classes_identical'] or
                maximum>audit['tolerance']):
            raise ValueError(f'Inisialisasi S4 tidak mereproduksi S3: {audit}')
        return audit,prediction if not full else None

    @torch.inference_mode()
    def predict(self,model,rows,scaler,cached):
        model.eval(); ys=[]; probabilities=[]; global_p=[]; beat_y=[]; beat_p=[]
        for batch in self.loader(rows,scaler,cached):
            batch=to_device(batch,self.device); output=self.forward(model,batch)
            ys.extend(batch['y'].cpu()); probabilities.append(output['logp'].exp().cpu().numpy())
            global_p.append(output['global_logits'].softmax(-1).cpu().numpy())
            valid=batch['mask'] & (batch['beat']>=0) & (batch['y'][:,None]<3)
            beat_y.extend(batch['beat'][valid].cpu()); beat_p.append(output['beat_logits'][valid].softmax(-1).cpu().numpy())
        return {'y':np.asarray(ys),'prob':np.concatenate(probabilities),'global_prob':np.concatenate(global_p),
            'beat_y':np.asarray(beat_y),'beat_prob':np.concatenate(beat_p) if beat_p else np.empty((0,3))}

    def fit(self,fold,full=False,epochs=None):
        folder=self.run_dir(fold,full); folder.mkdir(parents=True,exist_ok=True)
        result_path=folder/'result.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if (result.get('protocol_hash') != self.p.hash or
                    result.get('method_hash') != method_hash() or
                    digest(folder/'best.pt') != result['checkpoint_sha256']):
                raise ValueError('Run lengkap berbeda protokol/metode atau checkpoint rusak.')
            if full and self.scenario in ('S2','S3') and not (folder/'initialization_anchor.npz').exists():
                raise ValueError('Final frozen model tidak memiliki initialization anchor.')
            if self.scenario=='S4' and not (folder/'initialization_audit.json').exists():
                raise ValueError('Run S4 tidak memiliki audit inisialisasi S3.')
            return result
        train=self.dev if full else self.dev[self.dev.fold!=int(fold)].reset_index(drop=True)
        valid=None if full else self.dev[self.dev.fold==int(fold)].reset_index(drop=True)
        if valid is not None and set(train.patient_id)&set(valid.patient_id): raise ValueError('Overlap pasien.')
        seed=int(self.p.cfg['seed']+(100 if full else int(fold))); seed_all(seed,self.cfg['deterministic'])
        last_path=folder/'last.pt'
        model,scaler,cached=self._model_scaler(train,fold,full,last_path.exists())
        s4_initial_prediction=None
        initialization_audit=folder/'initialization_audit.json'
        if self.scenario=='S4' and not last_path.exists():
            audit,s4_initial_prediction=self._audit_s4_initialization(
                model,scaler,fold,full,valid)
            atomic_json(initialization_audit,audit)
        elif self.scenario=='S4' and last_path.exists() and not initialization_audit.exists():
            raise ValueError('Checkpoint S4 tidak memiliki audit inisialisasi S3.')
        weights,counts=self.base.weights(train,TAU[self.scenario])
        self._active_weights={key:value.detach().cpu().tolist() for key,value in weights.items()}
        optimizer=self._optimizer(model)
        max_epochs=int(self.cfg['max_epochs'] if epochs is None else epochs)
        if max_epochs < 0:
            raise ValueError('Jumlah epoch tidak boleh negatif.')
        # Anggaran epoch final boleh lebih pendek dari batas CV, tetapi lintasan
        # learning-rate harus tetap sama dengan CV pada epoch yang bersesuaian.
        # Memakai T_max=max_epochs final akan mempercepat decay dan membuat hasil
        # CV tidak lagi menjadi dasar yang setara untuk memilih epoch final.
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,T_max=max(1,int(self.cfg['max_epochs'])))
        amp=torch.amp.GradScaler('cuda',enabled=self.cfg['amp'] and self.device.type=='cuda')
        history=[]; best=-1.; best_epoch=0; start=0; elapsed=0.
        if last_path.exists():
            ck=load_checkpoint(last_path)
            if ck['method_hash']!=method_hash() or ck['max_epochs']!=max_epochs: raise ValueError('Checkpoint resume berbeda.')
            model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler']); amp.load_state_dict(ck['amp'])
            history,best,best_epoch,start,elapsed=ck['history'],ck['best'],ck['best_epoch'],ck['epoch'],ck['elapsed']
            torch.set_rng_state(ck['rng_torch']); np.random.set_state(ck['rng_numpy']); random.setstate(ck['rng_python'])
            if self.device.type=='cuda': torch.cuda.set_rng_state_all(ck['rng_cuda'])
        elif self.scenario=='S4' and valid is not None:
            prediction=s4_initial_prediction
            best=calculate(prediction['y'],prediction['prob'])[0]['macro_f1']
            history=[{'epoch':0,'train_loss':0.,'val_macro_f1':best,'elapsed_s':0.}]
            payload=self._payload(model,optimizer,scheduler,amp,scaler,train,fold,full,0,max_epochs,history,best,0,elapsed,counts)
            atomic_torch(folder/'best.pt',payload); atomic_torch(last_path,payload)
        if max_epochs==0:
            payload=self._payload(model,optimizer,scheduler,amp,scaler,train,fold,full,0,max_epochs,history,best,0,elapsed,counts)
            atomic_torch(folder/'best.pt',payload); atomic_torch(last_path,payload)
        for epoch in range(start,max_epochs):
            if valid is not None and epoch-best_epoch>=self.cfg['patience']: break
            begin=time.perf_counter(); model.train(); optimizer.zero_grad(set_to_none=True)
            loader=self.loader(train,scaler,cached,True,seed+epoch); losses=0.; windows=0
            for step,batch in enumerate(tqdm(loader,desc=f'{self.arch} {self.scenario} fold {fold} epoch {epoch+1}',leave=False)):
                batch=to_device(batch,self.device)
                group_start=(step//self.cfg['accumulation_steps'])*self.cfg['accumulation_steps']
                group_windows=min(self.cfg['accumulation_steps']*loader.batch_size,len(train)-group_start*loader.batch_size)
                with torch.autocast(self.device.type,enabled=amp.is_enabled()):
                    output=self.forward(model,batch); loss,_=loss_terms(output,batch,weights,self.cfg)
                if not torch.isfinite(loss): raise FloatingPointError('Loss tidak finite.')
                amp.scale(loss*len(batch['y'])/group_windows).backward(); losses+=float(loss.detach())*len(batch['y']); windows+=len(batch['y'])
                if (step+1)%self.cfg['accumulation_steps']==0 or step+1==len(loader):
                    amp.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); amp.step(optimizer); amp.update(); optimizer.zero_grad(set_to_none=True)
            prediction=None if valid is None else self.predict(model,valid,scaler,cached)
            score=None if prediction is None else calculate(prediction['y'],prediction['prob'])[0]
            scheduler.step(); elapsed+=time.perf_counter()-begin
            value=float(epoch+1) if score is None else score['macro_f1']; improved=value>best
            if improved: best,best_epoch=value,epoch+1
            history.append({'epoch':epoch+1,'train_loss':losses/windows,'val_macro_f1':None if score is None else value,'elapsed_s':elapsed})
            payload=self._payload(model,optimizer,scheduler,amp,scaler,train,fold,full,epoch+1,max_epochs,history,best,best_epoch,elapsed,counts)
            if improved: atomic_torch(folder/'best.pt',payload)
            atomic_torch(last_path,payload); pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
            print(f'{self.scenario} | fold {fold} | epoch {epoch+1} | macro-F1 {value:.4f}' if valid is not None else f'{self.scenario} final | epoch {epoch+1}/{max_epochs}')
        checkpoint=load_checkpoint(folder/'best.pt'); model.load_state_dict(checkpoint['model']); score={}
        if valid is not None:
            prediction=self.predict(model,valid,scaler,cached)
            score=export_metrics(folder,prediction['y'],prediction['prob'],valid,prediction['global_prob'],prediction['beat_y'],prediction['beat_prob'])
        elif self.scenario in ('S2','S3'):
            anchor_count=INITIALIZATION_ANCHOR_WINDOWS
            anchor_rows=self.dev.iloc[:anchor_count].reset_index(drop=True)
            anchor=self.predict(model,anchor_rows,scaler,cached)
            temporary=folder/'initialization_anchor.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,window_ids=np.asarray(
                    anchor_rows.window_id.astype(str).tolist(),dtype=np.str_),
                    labels=anchor['y'],probabilities=anchor['prob'])
            os.replace(temporary,folder/'initialization_anchor.npz')
        result={'arch':self.arch,'scenario':self.scenario,'fold':int(fold),'full':bool(full),'tau':TAU[self.scenario],
            'best_epoch':int(checkpoint['best_epoch']),'protocol_hash':self.p.hash,'method_hash':method_hash(),
            'checkpoint_sha256':digest(folder/'best.pt'),'training_s':elapsed,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),**score,
            'environment':environment(),'device':torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'cpu'}
        atomic_json(result_path,result); return result

    def _payload(self,model,optimizer,scheduler,amp,scaler,train,fold,full,epoch,max_epochs,history,best,best_epoch,elapsed,counts):
        return {'method_hash':method_hash(),'protocol_hash':self.p.hash,'arch':self.arch,'scenario':self.scenario,
            'tau':TAU[self.scenario],'fold':int(fold),'full':bool(full),'epoch':epoch,'max_epochs':max_epochs,
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'amp':amp.state_dict(),
            'rr_scaler':scaler.state_dict(),'history':history,'best':best,'best_epoch':best_epoch,'elapsed':elapsed,
            'counts':{k:v.tolist() for k,v in counts.items()},'weights':self._active_weights,'train_ids':train.window_id.tolist(),
            'rng_torch':torch.get_rng_state(),'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),
            'rng_cuda':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else []}

    def cv(self): return pd.DataFrame([self.fit(fold) for fold in range(5)])

    def select_final_epoch(self):
        """Pilih epoch yang memaksimalkan mean validation Macro-F1 kelima fold."""
        histories=[]
        fold_checkpoints={}
        for fold in range(5):
            folder=self.run_dir(fold)
            result=validate_completed_run(self.p,self.arch,self.scenario,fold,False)
            history=pd.read_csv(folder/'history.csv')
            history=history[np.isfinite(history.val_macro_f1)].copy()
            history['epoch']=history.epoch.astype(int)
            histories.append(history.set_index('epoch').val_macro_f1)
            fold_checkpoints[str(fold)]=result['checkpoint_sha256']
        common=set(histories[0].index)
        for history in histories[1:]: common &= set(history.index)
        if not common:
            raise ValueError('Tidak ada epoch valid yang tersedia pada seluruh fold.')
        rows=[]
        for epoch in sorted(common):
            values=[float(history.loc[epoch]) for history in histories]
            rows.append({'epoch':int(epoch),'mean_macro_f1':float(np.mean(values)),
                'std_macro_f1':float(np.std(values,ddof=1)),
                **{f'fold_{fold}_macro_f1':value for fold,value in enumerate(values)}})
        curve=pd.DataFrame(rows)
        best=float(curve.mean_macro_f1.max())
        tolerance=float(self.cfg.get('tie_tolerance',1e-4))
        selected=int(curve[curve.mean_macro_f1>=best-tolerance].epoch.min())
        receipt={'arch':self.arch,'scenario':self.scenario,
            'rule':'argmax mean validation Macro-F1 over epochs shared by all five folds',
            'tie_tolerance':tolerance,'selected_epoch':selected,
            'selected_mean_macro_f1':float(curve.loc[curve.epoch==selected,'mean_macro_f1'].iloc[0]),
            'common_epochs':curve.epoch.astype(int).tolist(),
            'fold_checkpoint_sha256':fold_checkpoints,
            'protocol_hash':self.p.hash,'method_hash':method_hash()}
        folder=self.p.out/'selection'/self.arch
        folder.mkdir(parents=True,exist_ok=True)
        curve.to_csv(folder/f'{self.scenario}_final_epoch_curve.csv',index=False)
        atomic_json(folder/f'{self.scenario}_final_epoch.json',receipt)
        return receipt

    def final(self):
        self.cv()
        selection=self.select_final_epoch()
        return self.fit(-1,True,int(selection['selected_epoch']))


def run_frozen(protocol,arch):
    base=Experiment(protocol,arch); cache=base.cache_features()
    rows=[]
    for scenario in ('S2','S3'):
        experiment=FinalExperiment(protocol,arch,scenario)
        rows.extend(experiment.cv().to_dict('records')); experiment.final()
    return cache,pd.DataFrame(rows)


def run_finetune(protocol,arch):
    experiment=FinalExperiment(protocol,arch,'S4')
    frame=experiment.cv(); experiment.final(); return frame


def validate_completed_run(protocol,arch,scenario,fold,full=False):
    """Validasi artefak, identitas checkpoint, dan cakupan train IDs satu run."""
    folder=protocol.out/('final_models' if full else 'cv')/arch/scenario/(
        'final' if full else f'fold_{int(fold)}')
    result_path=folder/'result.json'; checkpoint_path=folder/'best.pt'
    if not result_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f'Run belum lengkap: {folder}')
    result=json.loads(result_path.read_text())
    expected={'arch':arch,'scenario':scenario,'fold':-1 if full else int(fold),
        'full':bool(full),'protocol_hash':protocol.hash,'method_hash':method_hash()}
    for key,value in expected.items():
        if result.get(key)!=value:
            raise ValueError(f'Identitas result berbeda pada {folder}: {key}')
    if digest(checkpoint_path)!=result.get('checkpoint_sha256'):
        raise ValueError(f'Checksum checkpoint berbeda: {folder}')
    checkpoint=load_checkpoint(checkpoint_path)
    for key,value in expected.items():
        if checkpoint.get(key)!=value:
            raise ValueError(f'Identitas checkpoint berbeda pada {folder}: {key}')
    manifest=read_manifest(protocol.prepared/'global_manifest.csv')
    development=manifest[manifest.split=='development']
    expected_ids=set(development.window_id.astype(str) if full else
        development[development.fold!=int(fold)].window_id.astype(str))
    train_ids=list(map(str,checkpoint.get('train_ids',[])))
    if len(train_ids)!=len(set(train_ids)) or set(train_ids)!=expected_ids:
        raise ValueError(f'Cakupan train IDs berbeda: {folder}')
    if full and scenario in ('S2','S3') and not (folder/'initialization_anchor.npz').exists():
        raise FileNotFoundError(f'Initialization anchor tidak tersedia: {folder}')
    if scenario=='S4' and not (folder/'initialization_audit.json').exists():
        raise FileNotFoundError(f'Audit inisialisasi S4 tidak tersedia: {folder}')
    if not full:
        for name in ['metrics.json','per_class.csv','probabilities.npy','labels.npy','confusion.npy']:
            if not (folder/name).exists(): raise FileNotFoundError(folder/name)
    return result


def status(protocol):
    rows=[]
    for arch in DIMENSIONS:
        for scenario in SCENARIOS:
            for fold in range(5):
                path=protocol.out/'cv'/arch/scenario/f'fold_{fold}'/'result.json'
                rows.append({'arch':arch,'scenario':scenario,'fold':fold,'complete':path.exists()})
            path=protocol.out/'final_models'/arch/scenario/'final/result.json'
            rows.append({'arch':arch,'scenario':scenario,'fold':'final','complete':path.exists()})
    return pd.DataFrame(rows)


def export_bundle(protocol,arch,stage,destination):
    if stage not in ('s3','s4'): raise ValueError('Stage bundle harus s3/s4.')
    scenario=stage.upper(); roots=[protocol.out/'cv'/arch/scenario,protocol.out/'final_models'/arch/scenario]
    required=[roots[0]/f'fold_{i}/result.json' for i in range(5)]+[roots[1]/'final/result.json']
    if not all(path.exists() for path in required): raise RuntimeError(f'{stage} belum lengkap.')
    for fold in range(5): validate_completed_run(protocol,arch,scenario,fold,False)
    validate_completed_run(protocol,arch,scenario,-1,True)
    # last.pt berisi optimizer/resume state dan dapat menggandakan ukuran bundle.
    # Handoff hanya membutuhkan best.pt, result, metrik, dan riwayat final.
    files=[p for root in roots for p in root.rglob('*') if p.is_file() and p.name != 'last.pt']
    files += [protocol.out/'protocol_lock.json',protocol.out/'final_method_lock.json']
    manifest={'format':1,'arch':arch,'stage':stage,'protocol_hash':protocol.hash,'method_hash':method_hash(),
              'files':{p.relative_to(protocol.out).as_posix():digest(p) for p in files}}
    target=Path(destination); target.parent.mkdir(parents=True,exist_ok=True); temporary=target.with_suffix(target.suffix+'.tmp')
    with ZipFile(temporary,'w',ZIP_DEFLATED,allowZip64=True) as archive:
        for path in files: archive.write(path,path.relative_to(protocol.out).as_posix())
        archive.writestr('bundle_manifest.json',json.dumps(manifest,indent=2))
    os.replace(temporary,target); return {'bundle':str(target),'files':len(files),'sha256':digest(target)}


def import_bundle(protocol,bundle):
    with ZipFile(bundle) as archive:
        manifest=json.loads(archive.read('bundle_manifest.json'))
        if manifest['protocol_hash']!=protocol.hash or manifest['method_hash']!=method_hash(): raise ValueError('Bundle berbeda protokol/metode.')
        for name,expected in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('Path bundle tidak aman.')
            if hashlib.sha256(archive.read(name)).hexdigest()!=expected: raise ValueError(f'Checksum bundle berbeda: {name}')
        imported=0
        for name,expected in manifest['files'].items():
            target=protocol.out/Path(name)
            if target.exists():
                if digest(target)!=expected: raise FileExistsError(f'Artefak berbeda tidak ditimpa: {name}')
                continue
            target.parent.mkdir(parents=True,exist_ok=True); temporary=target.with_suffix(target.suffix+'.tmp')
            temporary.write_bytes(archive.read(name)); os.replace(temporary,target); imported+=1
    return {'bundle':str(bundle),'imported':imported,'stage':manifest['stage'],'arch':manifest['arch']}


def recap(protocol):
    rows=[]
    for arch in DIMENSIONS:
        for scenario in SCENARIOS:
            for fold in range(5):
                path=protocol.out/'cv'/arch/scenario/f'fold_{fold}'/'result.json'
                if not path.exists(): raise FileNotFoundError(f'Run belum lengkap: {path}')
                rows.append(validate_completed_run(protocol,arch,scenario,fold,False))
            final=protocol.out/'final_models'/arch/scenario/'final/result.json'
            if not final.exists(): raise FileNotFoundError(f'Final belum lengkap: {final}')
            validate_completed_run(protocol,arch,scenario,-1,True)
    frame=pd.DataFrame(rows); table=protocol.out/'tables'; table.mkdir(exist_ok=True)
    frame.to_csv(table/'cv_runs.csv',index=False)
    summary=frame.groupby(['arch','scenario'])[['macro_f1','accuracy','f1_S','training_s']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(x).rstrip('_') for x in summary.columns]; summary.to_csv(table/'cv_summary.csv',index=False)
    return summary


def evaluate_holdout(protocol):
    recap(protocol); manifest=read_manifest(protocol.prepared/'global_manifest.csv')
    test=manifest[manifest.split=='holdout'].reset_index(drop=True); development=manifest[manifest.split=='development'].reset_index(drop=True)
    roster={}
    for arch in DIMENSIONS:
        for scenario in SCENARIOS:
            folder=protocol.out/'final_models'/arch/scenario/'final'; meta=json.loads((folder/'result.json').read_text())
            roster[f'{arch}/{scenario}']=meta['checkpoint_sha256']
    roster_path=protocol.out/'holdout/model_roster.json'
    if roster_path.exists() and json.loads(roster_path.read_text())!=roster: raise ValueError('Roster holdout telah berubah.')
    atomic_json(roster_path,roster); rows=[]
    for arch in DIMENSIONS:
        for scenario in SCENARIOS:
            folder=protocol.out/'final_models'/arch/scenario/'final'; checkpoint=load_checkpoint(folder/'best.pt')
            output=protocol.out/'holdout'/arch/scenario; marker=output/'complete.json'
            if marker.exists():
                receipt=json.loads(marker.read_text())
                if (receipt.get('checkpoint_sha256') != roster[f'{arch}/{scenario}'] or
                        receipt.get('protocol_hash') != protocol.hash or
                        receipt.get('method_hash') != method_hash()):
                    raise ValueError(f'Receipt holdout berbeda: {arch}/{scenario}')
                for name, expected in receipt.get('files', {}).items():
                    if not (output/name).is_file() or digest(output/name) != expected:
                        raise ValueError(f'Artefak holdout rusak: {arch}/{scenario}/{name}')
                rows.append({'arch':arch,'scenario':scenario,**json.loads((output/'metrics.json').read_text())}); continue
            experiment=FinalExperiment(protocol,arch,scenario)
            scaler=FinalScaler(experiment.store,development.window_id)
            if scaler.state_dict()!=checkpoint['rr_scaler']: raise ValueError('Scaler final berbeda.')
            if scenario=='S4':
                head=experiment._head(); model=EncoderHeadNet(arch,protocol.cfg,head,2).to(experiment.device); model.load_state_dict(checkpoint['model'])
            else:
                head=experiment._head(); head.load_state_dict(checkpoint['model']); model=EncoderHeadNet(arch,protocol.cfg,head,0).to(experiment.device)
                encoder_audit,_=experiment._audit_raw_anchor(model,scaler,folder)
                atomic_json(output/'encoder_initialization_audit.json',encoder_audit)
            prediction=experiment.predict(model,test,scaler,False)
            score=export_metrics(output,prediction['y'],prediction['prob'],test,prediction['global_prob'])
            atomic_json(marker,{'checkpoint_sha256':roster[f'{arch}/{scenario}'],'protocol_hash':protocol.hash,
                'method_hash':method_hash(),'files':{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
            rows.append({'arch':arch,'scenario':scenario,**score})
    frame=pd.DataFrame(rows); frame.to_csv(protocol.out/'tables/holdout_summary.csv',index=False); return frame
