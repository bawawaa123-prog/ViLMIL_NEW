"""Local review assets. Default dry-run; optional manual selected-case attention replay."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from stage353_common import *


def omit_png_icc(data):
    """Copy PNG chunks, omitting only iCCP metadata; leave pixel chunks intact."""
    if data[:8]!=b'\x89PNG\r\n\x1a\n':
        raise ValueError('Expected PNG for ICC metadata fallback')
    chunks=[data[:8]];position=8;omitted=False
    while position<len(data):
        if position+12>len(data):
            raise ValueError('Truncated PNG chunk header')
        length=int.from_bytes(data[position:position+4],'big')
        end=position+12+length
        if end>len(data):
            raise ValueError('Truncated PNG chunk payload')
        kind=data[position+4:position+8]
        if kind==b'iCCP':
            omitted=True
        else:
            chunks.append(data[position:end])
        position=end
        if kind==b'IEND':
            if not omitted:
                raise ValueError('No PNG ICC chunk found for metadata fallback')
            return b''.join(chunks)
    raise ValueError('PNG missing IEND')


def load_patch_tile(path):
    """Read montage pixels; recover oversized ICC metadata in memory only."""
    import io
    from PIL import Image,ImageOps
    try:
        with Image.open(path) as image:
            tile=ImageOps.fit(image.convert('RGB'),(224,224))
        omitted=False
    except ValueError as exc:
        if str(exc)!='Decompressed Data Too Large':
            raise
        # Pillow bounds decompressed PNG metadata to 1 MiB. Image.convert('RGB')
        # does not apply ICC transforms; discard only iCCP for this montage read.
        # Do not relax global Pillow limits or allow truncated pixel data.
        with io.BytesIO(omit_png_icc(Path(path).read_bytes())) as stream:
            with Image.open(stream) as image:
                tile=ImageOps.fit(image.convert('RGB'),(224,224))
        omitted=True
    tile.info.clear()
    return tile,omitted


def attention_mean(weights, count):
    array=weights.detach().cpu().numpy()
    if array.shape[-1]!=count or not np.isfinite(array).all():
        raise ValueError('Unexpected prototype-to-patch attention shape')
    return array.reshape(-1,count).mean(axis=0)


def validate_report(root):
    manifest=read_json(root/'experiment_manifest.json')
    verify_hashes(root,manifest['output_sha256'])
    selection=pd.read_csv(root/'case_review_selection.csv')
    mapping=pd.read_csv(root/'private/slide_id_mapping.csv',dtype={'slide_id':str})
    if mapping.duplicated(['fold','case_token']).any() or selection.duplicated(['fold','case_token']).any():
        raise ValueError('Duplicate review identity')
    selected=selection.merge(mapping,on=['fold','case_token'],how='left',validate='one_to_one')
    if selected.slide_id.isna().any(): raise ValueError('Missing private identity mapping')
    return manifest,selected


def scatter(coords,values,title,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    artist=ax.scatter(coords[:,0],coords[:,1],c=values,s=12,cmap='viridis')
    ax.invert_yaxis(); ax.set_aspect('equal',adjustable='datalim')
    ax.set(title=title,xlabel='Stored X',ylabel='Stored Y')
    fig.colorbar(artist,ax=ax,label='Model score; not lesion truth')
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def extract_selected_attention(selected,inference_manifest,output,top_k):
    import torch
    from tools import stage335_routing_diagnostics as s335
    from tools import stage336_context_shuffle_stability as s336
    from utils.utils import get_simple_loader
    from types import SimpleNamespace
    # Reuse exporter filename resolver; never launch its unprotected legacy CLI.
    from export_top_patches_from_attention import patch_filename_candidates
    from PIL import Image,ImageDraw
    config=SimpleNamespace(task='task_adenocarcinoma',csv_path=str(matched.CSV),data_root_dir=str(matched.DATA),
        data_folder_s='features_biomedclip_5x',data_folder_l='features_biomedclip_20x',mapping_path=str(matched.MAPPING),
        text_prompt=None,text_prompt_path=str(matched.PROMPT))
    prompts=s335._prompts(config)
    dataset,_=s335._dataset(config,prompts)
    resources=[];edges=[];identities=[]
    for fold,part in selected.groupby('fold',sort=True):
        val=dataset.return_splits(from_id=False,csv_path=str(matched.SPLITS/f'splits_{fold}.csv'))[1]
        if not set(part.slide_id)<=set(val.slide_data.slide_id): raise ValueError('Review cases outside Validation')
        val.slide_data=val.slide_data[val.slide_data.slide_id.isin(part.slide_id)].reset_index(drop=True)
        loader=get_simple_loader(val,batch_size=1,num_workers=0,mode='transformer')
        by_id=part.set_index('slide_id')
        high=matched.collect('E0',int(fold));low=inspect_low(inference_manifest['low_run'],int(fold))
        if high['status']!='Complete' or low['status']!='Complete': raise ValueError('Review checkpoint unavailable')
        for arm,source,call_index,batch_coord,scale in [('L',low,0,1,'5x'),('H',high,1,3,'20x')]:
            s336._seed_everything(1)
            model=build_clean_model(source,prompts,s335.DEVICE)
            identities.append(dict(fold=int(fold),arm=arm,checkpoint=routing.checkpoint_identity(source['checkpoint'])))
            captured=[]
            def hook(module,inputs,result):
                captured.append(result[1].detach())
            handle=model.cross_attention_1.register_forward_hook(hook)
            try:
                with torch.no_grad():
                    for batch in loader:
                        sid=str(batch[5][0]); row=by_id.loc[sid];token=row.case_token
                        tensors=[x.to(s335.DEVICE) for x in batch[:5]]
                        captured.clear()
                        prob,_,_=model(*tensors,mapping=batch[6][0])
                        expected=np.array([row.low_p0,row.low_p1]) if arm=='L' else np.array([row.high_p0,row.high_p1])
                        if not np.allclose(prob[0].cpu().numpy(),expected,atol=1e-7,rtol=0):
                            raise ValueError('Review replay differs from independent prediction')
                        if len(captured)!=2: raise ValueError('Expected Low/High prototype attention calls')
                        coords_tensor=batch[batch_coord]
                        coords=(coords_tensor.squeeze(0) if coords_tensor.ndim==3 else coords_tensor).cpu().numpy()
                        scores=attention_mean(captured[call_index],len(coords))
                        chosen=np.argsort(-scores,kind='stable')[:top_k]
                        out=reserve(output/f'fold{fold}'/token/arm)
                        table=pd.DataFrame(dict(rank=np.arange(1,len(chosen)+1),patch_index=chosen,
                            x=coords[chosen,0],y=coords[chosen,1],attention=scores[chosen]))
                        table.to_csv(out/'top_attention_coordinates.csv',index=False)
                        # Bounded fixed spatial sample; top coordinates remain complete top_k.
                        indices=np.linspace(0,len(scores)-1,min(1024,len(scores)),dtype=int)
                        scatter(coords[indices],scores[indices],f'{token} {arm}: prototype attention sample',out/'attention_scatter.png')
                        tiles=[]
                        for rank,index in enumerate(chosen,1):
                            x,y=map(int,coords[index]);patchroot=matched.DATA/f'patches_{scale}'/sid
                            files=[patchroot/n for n in patch_filename_candidates(sid,x,y)]
                            path=next((p for p in files if p.is_file()),None)
                            resource=dict(fold=int(fold),case_token=token,arm=arm,rank=rank,
                                patch_index=int(index),x=x,y=y,status='Available' if path else 'Pending patch image',
                                patch_path=str(path) if path else '',icc_profile_omitted=False)
                            resources.append(resource)
                            if path:
                                tile,resource['icc_profile_omitted']=load_patch_tile(path)
                                ImageDraw.Draw(tile).text((4,4),f'{arm} #{rank}',fill='red');tiles.append(tile)
                            if arm=='H':
                                mapping=batch[6][0]
                                ptr=np.asarray(mapping['parent_ptr']);children=np.asarray(mapping['child_indices'])
                                for parent in range(len(ptr)-1):
                                    if int(index) in children[ptr[parent]:ptr[parent+1]]:
                                        edges.append(dict(fold=int(fold),case_token=token,high_patch_index=int(index),
                                            low_parent_index=parent,relation='Stored positive-overlap CSR; not pathology annotation'))
                        if tiles:
                            canvas=Image.new('RGB',(224*min(5,len(tiles)),224*((len(tiles)+4)//5)),color='white')
                            for i,tile in enumerate(tiles): canvas.paste(tile,((i%5)*224,(i//5)*224))
                            canvas.save(out/'top_patch_montage.png')
            finally:
                handle.remove();del model
                if s335.DEVICE.type=='cuda': torch.cuda.empty_cache()
    pd.DataFrame(resources).to_csv(output/'private/top_patch_resources.csv',index=False)
    pd.DataFrame(edges).to_csv(output/'same_region_mapping.csv',index=False)
    return identities


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--folds',type=int,nargs='+',default=list(range(5)))
    p.add_argument('--extract-attention',action='store_true',help='Manual selected-case inference; requires --execute')
    p.add_argument('--top-k',type=int,default=10)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    print('Python:',matched.environment())
    folds=validate_folds(args.folds)
    if not 1<=args.top_k<=32: p.error('--top-k must be 1..32')
    manifest,selected=validate_report(args.report_dir)
    selected=selected[selected.fold.isin(folds)]
    source=manifest['source_inference']
    from run_independent_inference import inspect_inference
    inference,state,_=inspect_inference(Path(source['run_dir']))
    if source['manifest_sha256']!=matched.digest(Path(source['run_dir'])/'experiment_manifest.json'):
        raise ValueError('Changed independent run')
    if any(state.loc[state.fold.isin(folds),'status']!='Complete'):
        raise ValueError('Selected folds are not complete')
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    print(f'Review plan: {len(selected)} cases; extract_attention={args.extract_attention}')
    if not args.execute:
        print('Dry-run only; no images, model or output.');return
    output=reserve(args.output_dir,private=True);reserve(output/'private',private=True)
    resources=[]
    for row in selected.itertuples():
        resources.append(dict(fold=row.fold,case_token=row.case_token,slide_id=row.slide_id,group=row.group,
            wsi_path=str(matched.DATA/'wsi'/f'{row.slide_id}.svs'),
            low_features=str(matched.DATA/'features_biomedclip_5x'/f'{row.slide_id}.h5'),
            high_features=str(matched.DATA/'features_biomedclip_20x'/f'{row.slide_id}.h5'),
            mapping=str(matched.MAPPING/f'{row.slide_id}.npz'),expert_review='Pending'))
    pd.DataFrame(resources).to_csv(output/'private/review_resources.csv',index=False)
    run=Path(inference['stage352_run'])
    for fold,part in selected.groupby('fold'):
        for filename,column in [('patch_samples.csv','gate'),('region_samples.csv','gate_mean')]:
            source_file=run/f'fold{fold}/E1/normal_seed0'/filename
            if not source_file.exists(): continue
            stats=pd.read_csv(source_file,dtype={'slide_id':str})
            for row in part.itertuples():
                s=stats[stats.slide_id==row.slide_id]
                if len(s):
                    out=output/f'fold{fold}'/row.case_token;out.mkdir(parents=True,exist_ok=True)
                    scatter(s[['x','y']].to_numpy(),s[column].to_numpy(),f'{row.case_token}: sampled Routing',out/f'{column}_routing_scatter.png')
    identities=extract_selected_attention(selected,inference,output,args.top_k) if args.extract_attention else []
    pd.DataFrame(selected.drop(columns=['slide_id'])).to_csv(output/'case_review_selection.csv',index=False)
    write_json(output/'figure_manifest.json',dict(status='Complete review assets; expert interpretation Pending',
        attention_status='Complete selected-case replay' if args.extract_attention else 'Pending optional selected-case replay',
        morphology_status='Pending expert review',checkpoint_identities=identities,
        report=str(args.report_dir.resolve()),report_manifest_sha256=matched.digest(args.report_dir/'experiment_manifest.json'),
        sha256={str(p.relative_to(output)):matched.digest(p) for p in output.rglob('*') if p.is_file()},
        note='Model scores/top patches are candidate attention patterns, not lesion truth. Existing patch montages only; no full WSI processing. Oversized PNG ICC metadata may be omitted in memory for RGB montage reads; recorded per patch, original files and pixel chunks unchanged.'))
    print('Local review assets:',output)


if __name__=='__main__':
    main()
