"""Download/decode LAW WebGraph inputs and construct the undirected LCC.

Java 17+ and a C++17 compiler are required. Large graphs remain outside Git.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = [
    ('it.unimi.dsi', 'webgraph', '3.6.12'), ('it.unimi.dsi', 'dsiutils', '2.7.4'),
    ('it.unimi.dsi', 'fastutil', '8.5.16'), ('it.unimi.dsi', 'sux4j', '5.4.1'),
    ('it.unimi.di', 'jsap', '20210129'), ('org.slf4j', 'slf4j-api', '2.0.3'),
    ('ch.qos.logback', 'logback-core', '1.3.4'), ('ch.qos.logback', 'logback-classic', '1.3.4'),
    ('net.sf.jung', 'jung-api', '2.1.1'), ('net.sf.jung', 'jung-io', '2.1.1'),
    ('org.apache.commons', 'commons-math3', '3.6.1'), ('commons-io', 'commons-io', '2.11.0')]


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=['in-2004', 'eu-2005', 'uk-2002'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--raw-dir', type=Path, help='Existing .graph, .properties and .md5sums files')
    p.add_argument('--jars', type=Path, help='Existing WebGraph dependency jars for offline operation')
    p.add_argument('--heap', default='8g')
    p.add_argument('--download-base', default='https://data.law.di.unimi.it/webdata',
                   help='WebGraph source base URL; downloaded files must pass the source checksums')
    p.add_argument('--no-proxy', action='store_true')
    p.add_argument('--save-npz', action='store_true', help='Also export NPZ (default for in-2004/eu-2005)')
    a = p.parse_args()
    if sys.byteorder != 'little':
        p.error('The native CSR pipeline requires a little-endian host')
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output.resolve()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if a.no_proxy else urllib.request.build_opener()

    def fetch(url, dest):
        part = dest.with_suffix(dest.suffix + '.partial')
        try:
            with opener.open(url, timeout=120) as source, part.open('wb') as target:
                shutil.copyfileobj(source, target)
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError('Download failed: ' + url + '; check network/provider availability, or supply --raw-dir and --jars for offline preparation') from error
        part.replace(dest)

    raw = a.raw_dir.resolve() if a.raw_dir else out / 'raw'
    if not a.raw_dir:
        raw.mkdir()
        for suffix in ['graph', 'properties', 'md5sums']:
            name = a.dataset + '.' + suffix
            fetch(a.download_base.rstrip('/') + '/' + a.dataset + '/' + name, raw / name)
    checks = {}
    for line in (raw / (a.dataset + '.md5sums')).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 32:
            checks[Path(parts[-1].lstrip('*')).name] = parts[0].lower()
    for suffix in ['graph', 'properties']:
        path = raw / (a.dataset + '.' + suffix)
        if path.name not in checks or digest(path, 'md5') != checks[path.name]:
            raise ValueError('Source checksum missing or mismatched: ' + path.name)
    jars = a.jars.resolve() if a.jars else out / 'jars'
    runtime = []
    if not a.jars:
        jars.mkdir()
        for group, name, version in ARTIFACTS:
            filename = f'{name}-{version}.jar'
            url = f'https://repo.maven.apache.org/maven2/{group.replace(".", "/")}/{name}/{version}/{filename}'
            path = jars / filename
            fetch(url, path)
            with opener.open(url + '.sha1', timeout=120) as source:
                expected = source.read().decode().split()[0].strip()
            if digest(path, 'sha1') != expected:
                raise ValueError('Jar checksum mismatch: ' + filename)
            runtime.append(dict(file=filename, source=url, sha256=digest(path)))
    else:
        for _, name, version in ARTIFACTS:
            path = jars / f'{name}-{version}.jar'
            runtime.append(dict(file=path.name, sha256=digest(path), origin='user-supplied'))
    build = out / 'build'
    build.mkdir()
    classpath = os.pathsep.join(str(jars / r['file']) for r in runtime)
    source = ROOT / 'src/preparation'
    subprocess.run(['javac', '-cp', classpath, '-d', str(build), str(source / 'ExportWebGraph.java')], check=True)
    subprocess.run(['java', '-Xmx' + a.heap, '-cp', str(build) + os.pathsep + classpath,
                    'ExportWebGraph', str(raw / a.dataset), str(out / 'edges.bin'), str(out / 'decode.json')], check=True)
    decode = json.loads((out / 'decode.json').read_text())
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    subprocess.run(compiler + ['-O3', '-std=c++17', str(source / 'convert_csr.cpp'), '-o', str(build / 'convert_csr')], check=True)
    processed = out / 'processed' / a.dataset
    subprocess.run([str(build / 'convert_csr'), str(decode['nodes']), str(out / 'edges.bin'), str(processed)], check=True)
    files = {name: digest(processed / name) for name in ['indptr.u64', 'indices.u32', 'original_ids.u32', 'graph.gxt']}
    if a.dataset != 'uk-2002' or a.save_npz:
        import numpy as np
        from scipy import sparse
        ptr = np.fromfile(processed / 'indptr.u64', dtype='<u8').astype(np.int64)
        ids = np.fromfile(processed / 'indices.u32', dtype='<u4').astype(np.int32)
        matrix = sparse.csr_matrix((np.ones(len(ids), bool), ids, ptr), shape=(len(ptr)-1, len(ptr)-1))
        sparse.save_npz(processed / (a.dataset + '.lcc.npz'), matrix)
        np.save(processed / (a.dataset + '.lcc.node_ids.npy'), np.fromfile(processed / 'original_ids.u32', dtype='<u4').astype(np.int64))
    (out / 'preparation.json').write_text(json.dumps(dict(dataset=a.dataset, decode=decode, processed_sha256=files,
        raw_sha256={a.dataset+'.'+s:digest(raw/(a.dataset+'.'+s)) for s in ['graph','properties']}, runtime=runtime), indent=2)+'\n')
    print('Prepared ' + str(processed))

if __name__ == '__main__':
    main()
