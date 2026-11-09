"""Download and prepare SNAP graphs without changing an existing input snapshot."""
from pathlib import Path
import argparse,json,sys,urllib.request,shutil,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from congestionlab.graphs import DATASETS,prepare_dataset,sha256

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',choices=sorted(DATASETS));p.add_argument('--output',type=Path,required=True);p.add_argument('--raw',type=Path);p.add_argument('--no-proxy',action='store_true');a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 catalog=json.loads((ROOT/'data/dataset_sources.json').read_text()) if (ROOT/'data/dataset_sources.json').exists() else {}
 expected=catalog.get(a.dataset,{})
 raw=a.raw
 if raw is None:
  raw=a.output/(a.dataset+'.txt.gz');partial=raw.with_suffix('.partial')
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({})) if a.no_proxy else urllib.request.build_opener()
  with opener.open(DATASETS[a.dataset]['url'],timeout=120) as source,partial.open('wb') as target:shutil.copyfileobj(source,target)
  partial.rename(raw)
 raw_hash=sha256(raw)
 if expected.get('raw_sha256') and raw_hash!=expected['raw_sha256']:raise ValueError('Raw checksum differs from the frozen source')
 meta=prepare_dataset(a.dataset,raw,a.output/'processed')
 for key in ['processed_nodes','processed_undirected_edges','node_ids_sha256','gxt_sha256']:
  if key in expected and meta.get(key)!=expected[key]:raise ValueError('Processed data mismatch: '+key)
 with zipfile.ZipFile(a.output/'processed'/meta['matrix_file']) as archive:
  members={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()}
 if expected.get('npz_member_sha256') and members!=expected['npz_member_sha256']:raise ValueError('Processed sparse matrix array mismatch')
 meta['npz_member_sha256']=members
 meta['npz_container_matches_reference']=meta['matrix_sha256']==expected.get('matrix_sha256')
 meta['raw_file']=raw.name;meta['source_status']='Prepared from the public source.'
 (a.output/'processed'/f'{a.dataset}.lcc.metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
 (a.output/'validation.json').write_text(json.dumps({'dataset':a.dataset,'raw_sha256':raw_hash,'checked_keys':[k for k in ['raw_sha256','processed_nodes','processed_undirected_edges','node_ids_sha256','gxt_sha256','npz_member_sha256'] if k in expected],'npz_container_matches_reference':meta['npz_container_matches_reference'],'note':'NPZ compression bytes may differ between zlib builds; uncompressed array, GXT and original-ID hashes are checked.','processed_nodes':meta['processed_nodes'],'processed_undirected_edges':meta['processed_undirected_edges']},indent=2)+'\n')
 print(json.dumps({'dataset':a.dataset,'vertices':meta['processed_nodes'],'edges':meta['processed_undirected_edges'],'matrix_sha256':meta['matrix_sha256']}))
if __name__=='__main__':main()
