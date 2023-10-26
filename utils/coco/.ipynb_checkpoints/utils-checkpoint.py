from pathlib import Path


def get_files(path, extensions, relative=True):

    if type(extensions) is str:
        extensions = [extensions]

    all_files = []
    for ext in extensions:
        if relative:
            all_files.extend(Path(path).rglob(ext))
        else:
            all_files.extend(Path(path).glob(ext))

    for i in range(len(all_files)):
        all_files[i] = str(all_files[i])

    return all_files
