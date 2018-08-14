# coding=utf-8

from IPython.core.interactiveshell import InteractiveShell
import os
# ------------------------------------------------------------
from tensorflow.python.client import device_lib
import tensorflow as tf
# ------------------------------------------------------------
import urllib.request
import tarfile


# ------------------------------------------------------------

def get_available_gpus():
	local_device_protos = device_lib.list_local_devices()
	[
	print(x.name)
	for x in local_device_protos]
	return [x.name for x in local_device_protos]


def showAllVariables(config=True):
	if config == True:
		InteractiveShell.ast_node_interactivity = "all"
	else:
		InteractiveShell.ast_node_interactivity = "last_expr"


def init():
	get_available_gpus()
	os.environ['CUDA_VISIBLE_DEVICES'] = "0"
	print('Start using first GPU')


def limitGPUByrate(rate):
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = rate
	Sess = tf.Session(config=config)
	return Sess


def limitGPUByGrowth():
	config = tf.ConfigProto()
	gpu_options = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_options)
	Sess = tf.Session(config=config)
	return Sess

download_progress = 0
def downLoadDataSet(url, downloadPath):

	def report(block_no, block_size, file_size):  # list download progress
		global download_progress
		download_progress += block_size
		if (block_no % 500 == 0) or (download_progress == file_size):
			print("Downloaded block %i, %i/%i bytes recieved." % (block_no, download_progress, file_size))

	if not os.path.isfile(downloadPath):
		result, headers = urllib.request.urlretrieve(url, downloadPath, reporthook=report)
		print("Download complete, saved as %s" % (result))


def extractallTarFile(downloadPath,extraFolder,extraPath):
	if not os.path.exists(extraPath):
		tfile = tarfile.open(downloadPath, 'r:gz')  # tfile object
		result = tfile.extractall(extraFolder)  # extract to target folder
		return result