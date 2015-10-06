# -*- coding: utf-8 -*-
import subprocess
import time
from subprocess import Popen, PIPE
import tempfile
import threading
import datetime
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
from flask import Flask, render_template, request, jsonify
import logging

logging.basicConfig(filename='error.log', level=logging.DEBUG)


# Initialize the Flask application
app = Flask(__name__)


class operationTracker():
	def __init__(self):
		self.readOperationArray = []
		self.writeOperationArray = []
		
	def setOpArray(self, reads, writes):
		self.readOperationArray.append(reads)
		self.writeOperationArray.append(writes)
		return

class UserInput:
	def __init__(self):
		self.dbNumber = 0
		self.dbSelected = ""
		self.instances = 0
		self.targetDevice = 0
		self.keySize = ""
		self.flags = " -f "
	
	## modified from printMenu
	def setSelectedDB(self, selected):
		self.dbNumber = selected
		return 
		
	def getDatabase(self, selection):
		if(selection == 0):
			benchmark = './fdb_bench'
		if(selection == 1):
		     benchmark = './wt_bench'
		if(selection == 2):
			benchmark = './leveldb_bench'
		if (selection == 3):
			benchmark = './rocksdb_bench'
		self.dbSelected = benchmark
		return 
	
	
	def getInstances(self, threads):
		self.instances = threads
		return
		
	def setTarget(self, target):
		self.targetDevice = target
		return 
		
	def setkeySize(self, key):
		self.keySize = key
	
	def getKeySize(self, pattern):
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
		
		self.setkeySize(ini)
		return

class fileManager:
	def __init__(self, UserInput):
		self.dbFilePaths = []
		self.pattern = "filename = data/dummy"
		self.logPath = "filename = logs/ops_log"	
		
	def createFiles(self, dbSelected, target, instance):
		dbPaths = []
		devices = []
		
		if(dbSelected == 0):
			dbase = 'fdb'
		else:
			dbase = 'wt'
			
		if (target == 0):
			devices.append("/mnt/nvme/data_" + dbase)
		if (target == 1):
			devices.append("/mnt/sata/data_" + dbase)
		if (target == 2):
			devices.append("/mnt/sas/data_" + dbase)
		if (target == 3):
			devices.append("/mnt/ssdh/data_" + dbase)
		if (target == 4):
			devices.append("/mnt/nvme/data_" + dbase)
			devices.append("/mnt/sata/data_" + dbase)
			devices.append("/mnt/sas/data_" + dbase)
			devices.append("/mnt/ssdh/data_" + dbase)
				
		for i in range(instance):
			for dev in devices:
				path = dev + str(i)
				self.dbFilePaths.append(path + '/dummy')
		return
	
	def replace(self, nf_handle, source_ini, pattern, substr):
		with open(nf_handle, 'w') as new_file:
			with open(source_ini, 'r') as old_file:
				for line in old_file:
					new_line = line.replace(pattern, substr)
					new_file.write(new_line)
		return nf_handle
	
	def update(self, file_path, pattern, subst):
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
	
	def createIni(self, keySize, dbFilePaths):
		fileHandles = []
		
		for path in dbFilePaths:
			fHandle = tempfile.NamedTemporaryFile(delete=False)
			dummyPath = "filename =  " + path
			fHandle = self.replace(fHandle.name, keySize, self.pattern, dummyPath)
			fileHandles.append(fHandle)
		self.dbFilePaths = fileHandles
		return
	
	def updateLogfilePath(self, dbFilePaths):
		i = 0
		for ini in dbFilePaths:
			self.update(ini, self.logPath, self.logPath + str(i))
			i = i + 1
		return

class threader:
		def __init__(self):
			self.threads = []
		
		def benchMark(self, bench, fileName, flags):
			command = bench + flags + fileName
			p = subprocess.Popen([command], shell=True) 
			p.communicate()
			return 

		def bundleThreads(self, fileHandles, dbSelected, keySize, flags):
			threads = []
			for fileName in fileHandles:
				t = threading.Thread(target=self.benchMark, args=(dbSelected, fileName, flags))
				t.start()
				threads.append(t)
				
			for thread in threads:
				thread.join()
			return
			
class processData:

	def __init__(self):
		self.logFiles = []
		self.dataSet = []
		self.avgData = []
		self.statistics = []
		self.read_opsArr = []
		self.write_opsArr = []
	
	def findLastLogs(self, numInstances):
		files = []
		files = glob.glob('./logs/*')
		files.sort(key=os.path.getmtime)
		files.reverse()
		files = files[:(numInstances)]
		self.logFiles = files
		return

	def getEngine(self, engine):
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
		
	def getDevice(self, device):
		if (device == 'nvme'):
			test = 0
		if (device == 'sata'):
			test = 1
		if (device == 'sas'):
			test = 2
		if (device == 'ssdh'):
			test = 3

		return test
	
	def ss_norm(self, norm):
		regex = re.compile('[Norm]')  # etc.
		stripped_norm = regex.sub('', norm)
		norm_dist = stripped_norm.split(',')
		return norm_dist
	
	# Strip parathesis, if a num convert to float
	def strip(self, line):
		number = re.sub('[()]', '', line)

		return number

		
	def getStats(self, stats):
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
				statistics['engine'] = self.getEngine(self.strip(split_line[2]))
			if (line_num == 1):
				statistics['device'] = self.getDevice(self.strip(split_line[1].split('/')[2]))
			if (line_num == 2):
				statistics['documents'] = self.strip(split_line[6])
			if (line_num == 3):
				statistics['thread_reader'] = re.sub('[\W_]+', '', (self.strip(split_line[3])))
				statistics['thread_writer'] = re.sub('[\W_]+', '', self.strip(split_line[7]))
			if (line_num == 4):
				statistics['cache_size'] = self.strip(split_line[3])
			if (line_num == 5):
				statistics['key_len'] = self.ss_norm(self.strip((split_line[2])))[0]
				statistics['key_len_dist'] = self.ss_norm(self.strip(split_line[2]))[1]
				## 
				statistics['body_len'] = self.ss_norm(self.strip(split_line[6]))[0]
				statistics['body_len_dist'] = self.ss_norm(self.strip(split_line[6]))[1]
			if (line_num == 6):
				statistics['duration'] = self.strip(split_line[2])
			if (line_num == 7):
				statistics['reads'] = self.strip(split_line[0])
				statistics['read_ops'] = self.strip(split_line[2])
				statistics['read_us'] = self.strip(split_line[4])
			if (line_num == 8):
				statistics['writes'] = self.strip(split_line[0])
				statistics['write_ops'] = self.strip(split_line[2])
				statistics['write_us'] = self.strip(split_line[4])
			if (line_num == 9):
				statistics['total_ops'] = self.strip(split_line[1])
			if (line_num == 10):
				statistics['bytes_written'] = self.strip(split_line[2])
				statistics['GB_written'] = self.strip(split_line[5])
			if (line_num == 11):
				statistics['bbytes_written'] = self.strip(split_line[1])
				statistics['bGB_written'] = self.strip(split_line[3])
			if (line_num == 12):
				statistics['avg_write_put'] = self.strip(split_line[4])
			if (line_num == 13):
				statistics['written_perdoc'] = self.strip(split_line[0])
				statistics['write_amp'] = self.strip(split_line[6])
			line_num = line_num + 1
		
		return statistics
	
	def readStatLines(self, log_file):
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
   
   

		
	
		
	def prepareData(self):
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
		for key_dict in self.dataSet:
			for key, val in key_dict.iteritems():
				value = float(val)
				avg[key] += (value) / len(self.dataSet)
		self.avgData = avg	
		return
	
	def arrPlot(self):
		self.write_opsArr = self.avgData['write_ops']
		self.read_opsArr = self.avgData['read_ops']
		pprint.pprint(self.write_opsArr)
		pprint.pprint(self.read_opsArr)
		return

class operationTracker():
	def __init__(self):
		self.readOperationArray = []
		self.writeOperationArray = []
		
	def setOpArray(self, reads, writes):
		self.readOperationArray.append(reads)
		self.writeOperationArray.append(writes)
		return
		
class plotter():
	def __init__(self):
		self.my_string = "Take us to your leader!"
		
	def plotGraph(self, figure, dataSetOne, dataSetTwo, targetDevice):
		if(len(dataSetOne) != len(dataSetTwo)):
			print "Error: The datasets are of uneven lengths!"
			return -1
		
		N = len(dataSetOne)
		ind = np.arange(N)
		width = 0.35
		fig, ax = plt.subplots()
		rect1 = ax.bar(ind, dataSetOne, width, color='b')
		rect2 = ax.bar(ind + width, dataSetTwo, width, color='r')
		drive_name = ['Intel P3700', 'Intel S3710', 'Seagate SAS', 'Drive Space']
		ax.set_xticks(ind + width)
		ax.set_xticklabels(('NVMe', drive_name[targetDevice]))
		self.autolabel(rect1)
		self.autolabel(rect2)
		plt.savefig(figure)
		
	def autolabel(self, rects):
		for rect in rects:
			height = rect.get_height()
			if (height > 0):
				plt.text(rect.get_x() + rect.get_width() / 2., 1.015 * height, '%d' % (height),
						ha='center')



@app.route('/')
def index():
	return render_template('index.html')


# Route that will process the AJAX request
@app.route('/echo/')
def echo():
	### Get the database engine to run
	engine = request.args.get('engine', 0, type=int)
	### Get the drive to target
	drive = request.args.get('drive', 0, type=int)
	### Get the key,val patter to run 
	pattern = request.args.get('pattern', 0, type=int)
	### get the number of instances to run
	threads = request.args.get('instances', 0, type=int)
	print("Threads", threads);
	
	tracker = operationTracker()
	nvmeInput = UserInput()
	nvmeInput.setSelectedDB(engine)
	nvmeInput.setTarget(0)
	nvmeInput.getDatabase(nvmeInput.dbNumber)
	nvmeInput.getKeySize(pattern)
	nvmeInput.getInstances(threads)
	nvmeInput.flags = " -e -f "
	nvmeFileSet = fileManager(nvmeInput)
	nvmeFileSet.createFiles(nvmeInput.dbNumber, nvmeInput.targetDevice, nvmeInput.instances)
	pprint.pprint(nvmeFileSet.dbFilePaths)
	nvmeFileSet.createIni(nvmeInput.keySize, nvmeFileSet.dbFilePaths)
	nvmeFileSet.updateLogfilePath(nvmeFileSet.dbFilePaths)
	nvmeInstanceGrp = threader()
	nvmeInstanceGrp.bundleThreads(nvmeFileSet.dbFilePaths, nvmeInput.dbSelected, nvmeInput.keySize, nvmeInput.flags)
	nvmeDataCollection = processData()
	nvmeDataCollection.findLastLogs(nvmeInput.instances)
	nvmeDataCollection.collectData()
	nvmeDataCollection.prepareData()
	pprint.pprint(nvmeDataCollection.avgData)
	nvmeDataCollection.arrPlot()
	
	webInput = UserInput()
	webInput.setSelectedDB(engine)
	webInput.setTarget(drive)
	webInput.getDatabase(webInput.dbNumber)
	webInput.getKeySize(pattern)
	webInput.getInstances(threads)
	webInput.flags = " -e -f "
	webFileSet = fileManager(webInput)
	webFileSet.createFiles(webInput.dbNumber, webInput.targetDevice, webInput.instances)
	webFileSet.createIni(webInput.keySize, webFileSet.dbFilePaths)
 	webFileSet.updateLogfilePath(webFileSet.dbFilePaths)
	webInstanceGrp = threader()
	webInstanceGrp.bundleThreads(webFileSet.dbFilePaths, webInput.dbSelected, webInput.keySize, webInput.flags)
	webDataCollection = processData()
	webDataCollection.findLastLogs(webInput.instances)
	webDataCollection.collectData()
	webDataCollection.prepareData()
 	pprint.pprint(webDataCollection.avgData)
	webDataCollection.arrPlot()
	
	tracker.setOpArray(nvmeDataCollection.write_opsArr, nvmeDataCollection.read_opsArr)
	tracker.setOpArray(webDataCollection.write_opsArr, webDataCollection.read_opsArr)

	dt = str(datetime.datetime.now()).replace(" ", "").replace(":", "").replace(".", "")
	figure = './static/graph' + dt + '.png'
		
	testPlot = plotter()
	testPlot.plotGraph(figure, tracker.readOperationArray,tracker.writeOperationArray, webInput.targetDevice)
	
	return jsonify(response=figure)


if __name__ == '__main__':
	app.run()

