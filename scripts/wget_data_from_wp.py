import wget
import numpy as np
from tqdm import trange
# from Samantha Huntington


def get_files(station: str):
    """wget download for waterproperties files"""

    path = f"E:\\charles\\mooring_data_page\\{station.lower()}\\"
    address = np.genfromtxt(path + f"wget_file_download_list_{station.lower()}.csv", dtype=str)

    for i in trange(address.size):
        wget.download('https://' + address[i], out=path + 'ios_shell_data\\')
    return
