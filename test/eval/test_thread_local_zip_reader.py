import concurrent.futures
import zipfile

from lmms_eval.tasks._task_utils.zip_reader import ThreadLocalZipReader


def test_thread_local_zip_reader_supports_concurrent_reads(tmp_path):
    zip_path = tmp_path / "images.zip"
    payloads = {
        f"images/{idx:03d}.txt": (f"payload-{idx}-" * 2048).encode()
        for idx in range(12)
    }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)

    reader = ThreadLocalZipReader(lambda: zip_path)
    names = list(payloads)

    def read_one(index):
        name = names[index % len(names)]
        return name, reader.read(name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(executor.map(read_one, range(512)))

    assert all(payload == payloads[name] for name, payload in results)
