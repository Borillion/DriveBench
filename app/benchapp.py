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
import matplotlib.lines as mlines

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
		self.newLogPath = "filename = /mnt/os/logs/ops_log"
	 
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
			dummyPath = "filename =	" + path
			fHandle = self.replace(fHandle.name, keySize, self.pattern, dummyPath)
			fileHandles.append(fHandle)
		self.dbFilePaths = fileHandles
		return
	
	def updateLogfilePath(self, dbFilePaths):
		i = 0
		for ini in dbFilePaths:
			self.update(ini, self.logPath, self.newLogPath + str(i))
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
		self.benchmarkData = []
	
	def findLastLogs(self, numInstances):
		files = []
		files = glob.glob('/mnt/os/logs/*')
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
	
	def ss_norm(self, norm):
		regex = re.compile('[Norm]')	# etc.
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

		# print("=====getStats=======")
		# pprint.pprint(stats)
		# print("======end getStats ======")
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
			'bGB_written': 0,
			'avg_write_put': 0,
			'written_perdoc': 0,
			'write_amp': 0
		}

		for line in stats:
			split_line = line.split()
			if (line_num == 0):
				print(line_num, split_line)
				statistics['engine'] = self.getEngine(self.strip(split_line[2]))
			if (line_num == 1):
				print(line_num, split_line)
				statistics['device'] = self.getDevice(self.strip(split_line[1].split('/')[2]))
			if (line_num == 2):
				print(line_num, split_line)
				statistics['documents'] = self.strip(split_line[6])
			if (line_num == 3):
				print(line_num, split_line)
				statistics['thread_reader'] = re.sub('[\W_]+', '', (self.strip(split_line[3])))
				statistics['thread_writer'] = re.sub('[\W_]+', '', self.strip(split_line[7]))
			if (line_num == 4):
				print(line_num, split_line)
				statistics['cache_size'] = self.strip(split_line[3])
			if (line_num == 5):
				print(line_num, split_line)
				statistics['key_len'] = self.ss_norm(self.strip((split_line[2])))[0]
				statistics['key_len_dist'] = self.ss_norm(self.strip(split_line[2]))[1]
				statistics['body_len'] = self.ss_norm(self.strip(split_line[6]))[0]
				statistics['body_len_dist'] = self.ss_norm(self.strip(split_line[6]))[1]
			if (line_num == 6):
				print(line_num, split_line)
				statistics['duration'] = self.strip(split_line[2])
			if (line_num == 7):
				print(line_num, split_line)
				statistics['reads'] = self.strip(split_line[0])
				statistics['read_ops'] = self.strip(split_line[2])
				statistics['read_us'] = self.strip(split_line[4])
			if (line_num == 8):
				print(line_num, split_line)
				statistics['writes'] = self.strip(split_line[0])
				statistics['write_ops'] = self.strip(split_line[2])
				statistics['write_us'] = self.strip(split_line[4])
			if (line_num == 9):
				print(line_num, split_line)
				statistics['total_ops'] = self.strip(split_line[1])
			if (line_num == 10):
				print(line_num, split_line)
				statistics['bytes_written'] = self.strip(split_line[1])
				statistics['GB_written'] = self.strip(split_line[3])
			if (line_num == 11):
				print(line_num, split_line)
				statistics['bbytes_written'] = self.strip(split_line[2])
				statistics['bGB_written'] = self.strip(split_line[5])
			if (line_num == 12):
				print(line_num, split_line)
				statistics['avg_write_put'] = self.strip(split_line[4])
			if (line_num == 13):
				print(line_num, split_line)
				statistics['written_perdoc'] = self.strip(split_line[0])
				statistics['write_amp'] = self.strip(split_line[6])
			line_num = line_num + 1
		
		return statistics
	
	def readStatLines(self, log_file):

		stats = [] 
		logFile = []
		indices = []
		
		configList = [
			"DB module:",
			"filename: /mnt/",
			"# documents (i.e. working set size):",
			"# threads:",
			"block cache size:",
			"key length: Norm",
			"benchmark duration:"
			]
			
		elapsedStats = [
			"us/read",
			"us/write",
			"total",
			"bytes written",
			"written during benchmark",
			"average disk write throughput",
			"update"
			]

		n = 0
		with open(log_file) as file:
			for line in file:
				n += 1
				logFile.append(line)
				if(line == "\n"):
					indices.append(n)
	
		#print("=====logFile=======")
		#pprint.pprint(self.logFiles)
		#print("======end LogFile======")
		
		benchmarkConfig = logFile[indices[0]:indices[1]]
		benchmarkElapesedStats = logFile[indices[2]:indices[3]]

		benchmarkInstance = logFile[indices[1]:indices[2]][2:-1]
		writeLatency = logFile[indices[3]:indices[4]][2:-1]
		readLatency = logFile[indices[4]:indices[5]][2:-1]
		totalElapsed = logFile[indices[5]:]

		instanceData = []
		writeLatencyData = []
		readLatencyData = []	 
	 
		for x in benchmarkInstance:
			instanceData.append([float(item) for item in x.split()])
		self.benchmarkData.append(instanceData)
		
		for x in writeLatency:
			writeLatencyData.append([float(item) for item in x.split()])
		self.benchmarkData.append(writeLatencyData)
	 
		for x in readLatency:
			readLatencyData.append([float(item) for item in x.split()])
		self.benchmarkData.append(readLatencyData)
	 
		#pprint.pprint(writeLatency)
		#pprint.pprint(readLatency)
		#pprint.pprint(totalElapsed)

		del logFile
		del line
		
		
		for txt in configList:
			for line in benchmarkConfig:
				if(txt in line):
					location = benchmarkConfig.index(line)
					stats.append(benchmarkConfig[location])



		for txt in elapsedStats:
			for line in benchmarkElapesedStats:
				if ( (txt in line) and(line not in stats) ):
					location = benchmarkElapesedStats.index(line)
					stats.append(benchmarkElapesedStats[location])

		return stats
 
	 
	 
	def collectData(self):
		dataset = []
		for file in self.logFiles:
			stats = self.getStats(self.readStatLines(file))
			dataset.append(stats)
		print("==== READ STAT LINES ===")
		pprint.pprint(stats)
		print("==== * READ STAT LINES * ===")
		self.dataSet = dataset
		return


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

		
		N = len(dataSetOne)
		ind = np.arange(N)
		width = 0.35
		fig, ax = plt.subplots()
		rect1 = ax.bar(ind, dataSetOne, width, color='b')
		rect2 = ax.bar(ind + width, dataSetTwo, width, color='r')

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

	def plotGraphs(self, datasetCollection1, datasetCollection2, drive):
			
				"""Plot the reads and writes in a bar-chart"""
			
				drive_name = ['Intel P3700', 'Intel S3710', 'Intel S3710', 'Seagate SAS']
		
				fig = plt.figure()
				ax1 = plt.subplot2grid((12, 14), (0, 0), colspan=10, rowspan=3)
				ax1.set_title('NVMe')
				ax2 = plt.subplot2grid((12, 14), (3, 0), colspan=10, rowspan=3)
				ax3 = plt.subplot2grid((12, 14), (6, 0), colspan=10, rowspan=3)
				ax3.set_title(drive_name[drive])
				ax4 = plt.subplot2grid((12, 14), (9, 0), colspan=10, rowspan=3)
				
				elapsedTime1 = np.array(datasetCollection1.benchmarkData[0])[:,[0]].tolist()
				readOps1 = np.array(datasetCollection1.benchmarkData[0])[:,[3]].tolist()
				writeops1 = np.array(datasetCollection1.benchmarkData[0])[:,[4]].tolist()
				pprint.pprint(elapsedTime1)
				wLsample1 = np.array(datasetCollection1.benchmarkData[1])[:,[0]]
				wLval1 = np.array(datasetCollection1.benchmarkData[1])[:,[1]]	 
				 
				rLsample1 = np.array(datasetCollection1.benchmarkData[2])[:,[0]]
				rLval1 = np.array(datasetCollection1.benchmarkData[2])[:,[1]]	 
			
				a = ax1.plot(wLsample1, wLval1, color='g')
				b = ax2.plot(rLsample1, rLval1, color='m')
				ax1.tick_params(axis='both', labelsize=8)
				ax2.tick_params(axis='both', labelsize=8)
				
				aline_proxy = mlines.Line2D([], [], color='g')
				bline_proxy = mlines.Line2D([], [], color='m')
			
				leg = plt.legend([aline_proxy, bline_proxy], ['Write Latency', 'Read Latency'], loc='upper right', bbox_to_anchor=(1.47,6.02))
			
			
				# datasetCollection2
			
		
				elapsedTime2 = np.array(datasetCollection2.benchmarkData[0])[:,[0]].tolist()
				readOps2 = np.array(datasetCollection2.benchmarkData[0])[:,[3]].tolist()
				writeops2 = np.array(datasetCollection2.benchmarkData[0])[:,[4]].tolist()
				
				try:
						wLsample2 = np.array(datasetCollection2.benchmarkData[1])[:,[0]]
						wLval2 = np.array(datasetCollection2.benchmarkData[1])[:,[1]]	 
						c = ax3.plot(wLsample2, wLval2, color='g')
						ax3.tick_params(axis='both', labelsize=8)
				except Exception:
						pass			
		
				rLsample2 = np.array(datasetCollection2.benchmarkData[2])[:,[0]]
				rLval2 = np.array(datasetCollection2.benchmarkData[2])[:,[1]]	 
					
				d = ax4.plot(rLsample2, rLval2, color='m')
				ax4.tick_params(axis='both', labelsize=8)
		
				dt = str(datetime.datetime.now()).replace(" ", "").replace(".", "")
				graphFigure = './static/graph' + dt + '.png'
				#plt.subplots_adjust(hspace = 0.75)
				fig.tight_layout()
				fig.savefig(graphFigure)
				return graphFigure
	


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

	newPlot = plotter()
	graphFigure = newPlot.plotGraphs(nvmeDataCollection, webDataCollection, drive)
	pprint.pprint(graphFigure)
 
	return jsonify(response=graphFigure)


if __name__ == '__main__':
	app.run()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
