import requests, argparse



if (__name__ == "__main__"):

	parser = argparse.ArgumentParser(description="Downloading the Charm Dataset")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--id_dataset', type=int, help='Id of Dataset name')

	args = parser.parse_args()

	url_base = "https://repository.library.northeastern.edu/downloads/"

	url_list = ["neu:bz61jz54w?datastream_id=content", 
	"neu:bz61jz84m?datastream_id=content",
	"neu:bz61k080w?datastream_id=content",
	"neu:bz61k1472?datastream_id=content",
	"neu:bz61k148b?datastream_id=content",
	"neu:bz61k295x?datastream_id=content",
	"neu:bz61k316g?datastream_id=content",
	"neu:bz61k317r?datastream_id=content"]

	url_write_name_list = ["CLEAR", "LTE_1M", "LTE_FLOOD", "LTE_PINGs300", "LTE_ZT", "WIFI_1M", 
	"WIFI_FLOOD", "WIFI_PINGs300", "WIFI_ZT"]

	r = requests.get(url_base+url_list[args.id_dataset], allow_redirects=True)
	open(url_write_name_list[args.id_dataset]+".zip", 'wb').write(r.content)
