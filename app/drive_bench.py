# -*- coding: utf-8 -*-
import subprocess
import tempfile
import threading
import datetime
import time
import pprint
import re
import glob
import os
import pprint
from tempfile import mkstemp
from shutil import move
from os import remove, close
import numpy as np

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

### Devices ###
# nvme - nvme0 P3700
# sata - S3710 sdb
# sas - ST1200MM0017 sda
# ssdh - HITACHI HUSML4020ASS600 sdc 

#set the uid to nginx
#os.setuid(496)  
#os.setgid(496)

####
## variables 
#####
pattern = "filename = data/dummy"
log_path = "filename = logs/ops_log"

####
### Print menu
### returns - the selected benchmark value
####
def print_menu():
    print("")
    print("Select a Benchmark: ")
    print("0 - ForestDB")
    print("1 - Wired Tiger")
    print("2 - LevelDB")
    print("3 - RocksDB")
    user_selected = raw_input("Enter Value: ")
    return user_selected


####
### benchmark device destination
####
def print_location_menu():
    print("")
    print("Select at testing location: ")
    print("0 - NVMe Intel P3700")
    print("1 - SATA Intel S3710")
    print("2 - SAS Seagate HDD")
    print("3 - SAS SSD Hitachi")
    print("4 - All Drives")
    location = raw_input("Enter Value: ")
    return location


####
### Ask pattern type
####
def get_pattern():
    print ("")
    print ("Select a benchmark type: ")
    print ("0 - Default")
    print ("1 - 48")
    print ("2 - 256")
    print ("3 - 1024")
    print ("4 - Write Heavy")
    print ("5 - Write Only")
    pattern = int(raw_input("Enter Value: "))

    if (pattern == 0):
        ## default
        ini = './configs/default/bench_config.ini'
    if (pattern == 1):
        ini = './configs/48/bench_config.ini'
    if (pattern == 2):
        ini = './configs/256/bench_config.ini'
    if (pattern == 3):
        ini = './configs/1024/bench_config.ini'
    if (pattern == 4):
        ini = './configs/writeheavy/bench_config.ini'
    if (pattern == 5):
        ini = './configs/writeonly/bench_config.ini'
    print(ini)
    return ini


####
### Threads
### return the number concurrent benchmarks
####
def print_thread_menu():
    print("")
    print("Enter number of instances to run [1 - 9]: ")
    threads = raw_input("Enter Value: ")
    return threads


####
### Set the executable for the benchmark
####
def select_benchmark(selection):
    if (selection == 0):
        benchmark = './fdb_bench'
    if (selection == 1):
        benchmark = './wt_bench'
    if (selection == 2):
        benchmark = './leveldb_bench'
    if (selection == 3):
        benchmark = './rocksdb_bench'
    return benchmark


def create_files(selection, location, threads):
    # Array of database paths
    db_paths = []
    devices = []
    
    print(selection)
    
    if (selection == 0):
        dbase = 'fdb'
    else:
        dbase = 'wt'

    # select device to create db path on
    if (location == 0):
        devices.append("/mnt/nvme/data_" + dbase)
    if (location == 1):
        devices.append("/mnt/sata/data_" + dbase)  # S3710 sda
    if (location == 2):
        devices.append("/mnt/sas/data_" + dbase)
    if (location == 3):
        devices.append("/mnt/ssdh/data_" + dbase)
    if (location == 4):
        devices.append("/mnt/nvme/data_" + dbase)
        devices.append("/mnt/sata/data_" + dbase)
        devices.append("/mnt/sas/data_" + dbase)
        devices.append("/mnt/ssdh/data_" + dbase)

        # create a db path for each thread
    for i in range(threads):
        for dev in devices:
            path = dev + str(i)
            db_paths.append(path + '/dummy')
    # return the db paths
    return db_paths

####
### Replace string in file with another
####
def replace(nf_handle, source_ini, pattern, substr):
    with open(nf_handle, 'w') as new_file:
        with open(source_ini, 'r') as old_file:
            for line in old_file:
                new_line = line.replace(pattern, substr)
                new_file.write(new_line)
    return nf_handle


####
### Replace string in file with another
####
def update(file_path, pattern, subst):
    # Create temp file
    fh, abs_path = mkstemp()
    with open(abs_path, 'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    close(fh)
    # Remove original file
    remove(file_path)
    # Move new file
    move(abs_path, file_path)


####
### Create a configuration ini file for each
### of the iterated dummy file paths
### - returns an array of iterated file handles
####

def create_ini(source_ini, db_file_paths):
    # array of file tmpfs ini filehandles
    file_handles = []

    for path in db_file_paths:
        # make a tempfs file handle
        f_handle = tempfile.NamedTemporaryFile(delete=False)
        # match and replace db dummy file path
        dummy_path = "filename =  " + path
        f_handle = replace(f_handle.name, source_ini, pattern, dummy_path)
        file_handles.append(f_handle)
    return file_handles


def update_logfile_path(file_handles):
    i = 0

    for ini in file_handles:
        update(ini, log_path, log_path + str(i))
        i = i + 1
    return


##
# benchmark
##
def bench_mark(bench, filename):
    # s_out - string containing stdout from benchmark
    command = bench + ' -e -f ' + filename
    print(filename)
    p = subprocess.Popen([command], shell=True) 
    # p.wait()
    p.communicate()
    return


# Create and launch a thread
def threader(file_handles, benchmark):
    threads = []
    for filename in file_handles:
        t = threading.Thread(target=bench_mark, args=(benchmark, filename ))
        t.start()
        threads.append(t)

    for thread in threads:
        thread.join()
    return


##
# call secure erase
##
def call_SE():
    ##
    # What does this mean : SG_IO: questionable sense data, results may be incorrect
    ##
    w = subprocess.Popen(['umount', '/mnt/nvme0'])
    p = subprocess.Popen(['nvme', 'format', '/dev/nvme0', '--ses=1'])
    p.wait()
    q = subprocess.Popen(
        ['parted', '-s', '-a', 'optimal', '/dev/nvme0n1', 'mklabel', 'gpt', '--', 'mkpart', 'primary', 'ext4', '1 -1'])
    q.wait()
    r = subprocess.Popen(['mkfs.ext4', '/dev/nvme0n1p1'])
    r.wait()
    s = subprocess.Popen(['mount', '-t' 'ext4', '/mnt/nvme0', '/dev/nvme0n1p1'])
    s.wait()
    return


# Strip parathesis, if a num convert to float
def strip(line):
    number = re.sub('[()]', '', line)

    return number


def ss_norm(norm):
    regex = re.compile('[Norm]')  # etc.
    stripped_norm = regex.sub('', norm)
    norm_dist = stripped_norm.split(',')

    return norm_dist


def find_last_logs():
    last_run = []
    newest = max(glob.iglob('./logs/*'), key=os.path.getctime)
    name_strings = newest.split('_')
    for s in os.listdir('./logs/'):
        if name_strings[2] in s:
            last_run.append(s)
    return last_run


def read_stat_lines(log_file):
    list = [
        "DB module:",
        "filename: /mnt/",
        "# documents (i.e. working set size):",
        "# threads:",
        "block cache size:",
        "key length: Norm",
        "benchmark duration:",
        "us/read",
        "us/write",
        "total",
        "bytes written",
        "written during benchmark",
        "average disk write throughput",
        "written per doc update"]
    stats = []

    with open(log_file) as file:
        n = 0
        for line in file:
            if (n < len(list)):
                if (list[n] in line):
                    stats.append(line)
                    n = n + 1
    return stats


def get_engine(engine):
    test = 0
    if (engine == 'ForestDB'):
        test = 0
    if (engine == 'WiredTiger'):
        test = 1
    if (engine == 'LevelDB'):
        test = 2
    if (engine == 'RocksDB'):
        test = 3

    return test


def get_device(device):
    test = 0
    if (device == 'nvme'):
        test = 0
    if (device == 'sata'):
        test = 1
    if (device == 'sas'):
        test = 2
    if (device == 'ssdh'):
        test = 3

    return test


def get_stats(stats):
    line_num = 0
    split_line = []

    statistics = {
        'engine': 0,
        'device': 0,
        'documents': 0,
        'thread_reader': 0,
        'thread_writer': 0,
        'cache_size': 0,
        'key_len': 0,
        'key_len_dist': 0,
        'body_len': 0,
        'body_len_dist': 0,
        'duration': 0,
        'reads': 0,
        'read_ops': 0,
        'read_us': 0,
        'writes': 0,
        'write_ops': 0,
        'write_us': 0,
        'total_ops': 0,
        'bytes_written': 0,
        'GB_written': 0,
        'bbytes_written': 0,
        'bGB_written': 0,
        'avg_write_put': 0,
        'written_perdoc': 0,
        'write_amp': 0
    }

    for line in stats:
        split_line = line.split()
        if (line_num == 0):
            statistics['engine'] = get_engine(strip(split_line[2]))
        if (line_num == 1):
            statistics['device'] = get_device(strip(split_line[1].split('/')[2]))
        if (line_num == 2):
            statistics['documents'] = strip(split_line[6])
        if (line_num == 3):
            statistics['thread_reader'] = re.sub('[\W_]+', '', (strip(split_line[3])))
            statistics['thread_writer'] = re.sub('[\W_]+', '', strip(split_line[7]))
        if (line_num == 4):
            statistics['cache_size'] = strip(split_line[3])
        if (line_num == 5):
            statistics['key_len'] = ss_norm(strip((split_line[2])))[0]
            statistics['key_len_dist'] = ss_norm(strip(split_line[2]))[1]
            ## 
            statistics['body_len'] = ss_norm(strip(split_line[6]))[0]
            statistics['body_len_dist'] = ss_norm(strip(split_line[6]))[1]
        if (line_num == 6):
            statistics['duration'] = strip(split_line[2])
        if (line_num == 7):
            statistics['reads'] = strip(split_line[0])
            statistics['read_ops'] = strip(split_line[2])
            statistics['read_us'] = strip(split_line[4])
        if (line_num == 8):
            statistics['writes'] = strip(split_line[0])
            statistics['write_ops'] = strip(split_line[2])
            statistics['write_us'] = strip(split_line[4])
        if (line_num == 9):
            statistics['total_ops'] = strip(split_line[1])
        if (line_num == 10):
            statistics['bytes_written'] = strip(split_line[2])
            statistics['GB_written'] = strip(split_line[5])
        if (line_num == 11):
            statistics['bbytes_written'] = strip(split_line[1])
            statistics['bGB_written'] = strip(split_line[3])
        if (line_num == 12):
            statistics['avg_write_put'] = strip(split_line[4])
        if (line_num == 13):
            statistics['written_perdoc'] = strip(split_line[0])
            statistics['write_amp'] = strip(split_line[6])
        line_num = line_num + 1

    return statistics


def collect_data(logs):
    dataset = []
    for file in logs:
        stats = get_stats(read_stat_lines('./logs/' + file))
        dataset.append(stats)
    return dataset


def prepare_data(data):
    avg = {
        'engine': 0,
        'device': 0,
        'documents': 0,
        'thread_reader': 0,
        'thread_writer': 0,
        'cache_size': 0,
        'key_len': 0,
        'key_len_dist': 0,
        'body_len': 0,
        'body_len_dist': 0,
        'duration': 0,
        'reads': 0,
        'read_ops': 0,
        'read_us': 0,
        'writes': 0,
        'write_ops': 0,
        'write_us': 0,
        'total_ops': 0,
        'bytes_written': 0,
        'GB_written': 0,
        'bbytes_written': 0,
        'bGB_written': 0,
        'avg_write_put': 0,
        'written_perdoc': 0,
        'write_amp': 0
    }
    for key_dict in data:
        for key, val in key_dict.iteritems():
            value = float(val)
            avg[key] += (value) / len(data)
    return avg


def graph(data_1, data_2):
    # data arrays must be of the same number of elements
    if (len(data_1) != len(data_2)):
        return -1

    N = len(data_1)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    rect1 = ax.bar(ind, data_1, width, color='b')
    rect2 = ax.bar(ind + width, data_2, width, color='r')

    ax.set_xticks(ind + width)
    ax.set_xticklabels(('NVMe P3608', 'NVMe P3700'))

    autolabel(rect1)
    autolabel(rect2)

    dt = str(datetime.datetime.now()).replace(" ", "").replace(".", "")
    figure = 'graph' + dt + '.png'

    plt.savefig(figure)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if (height > 0):
            plt.text(rect.get_x() + rect.get_width() / 2., 1.015 * height, '%d' % (height),
                     ha='center')


### Ask user for what test they want to run
menu = int(print_menu())
### Select the appropriate benchmark
benchmark = select_benchmark(menu)
### Ask user for number of threads
num_threads = int(print_thread_menu())
### Select the config to run against
source_ini = get_pattern()
### Ask user to select device NVMe or SATA
location = int(print_location_menu()) ## set location to NVMe 
### Create db files paths based on selected NVMe or SATA
db_filepaths = create_files(menu, location, num_threads)
print pprint.pprint(db_filepaths)
###create ini files with iterated db dummy file paths
file_handles = create_ini(source_ini, db_filepaths)
update_logfile_path(file_handles)
print(file_handles)
print("Starting Instances!!\n")
threader(file_handles, benchmark)
logs = find_last_logs()
#call_SE()
data = collect_data(logs)
prep_data = prepare_data(data)

### Ask user for what test they want to run
menu_1 = int(print_menu())
### Select the appropriate benchmark
benchmark_1 = select_benchmark(menu_1)
### Ask user for number of threads
num_threads_1 = int(print_thread_menu())
### Select the config to run against
source_ini_1 = get_pattern()
### Ask user to select device NVMe or SATA
location_1 = int(print_location_menu()) ## set location to NVMe 
### Create db files paths based on selected NVMe or SATA 
db_filepaths_1 = create_files(menu_1, location_1, num_threads_1)
print pprint.pprint(db_filepaths_1)
###create ini files with iterated db dummy file paths
file_handles_1 = create_ini(source_ini, db_filepaths_1)
update_logfile_path(file_handles_1)
print(file_handles)
print("Starting Instances!!\n")
threader(file_handles_1, benchmark_1)
logs_1 = find_last_logs()
#call_SE()
data_nvme = collect_data(logs_1)
prep_data_nvme = prepare_data(data_nvme)

list = prep_data
list_1 = prep_data_nvme

data_1 = (float(list_1['write_ops']), float(list['write_ops']) )
data_2 = (float(list_1['read_ops']), float(list['read_ops']))
fig = graph(data_1, data_2) 

logs = find_last_logs()
#call_SE()
data = collect_data(logs)
pprint.pprint(data)
pprint.pprint (prep_data_nvme)
