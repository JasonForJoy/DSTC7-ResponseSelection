import numpy as np

def compute_recall(probs_aggre, labels_aggre):
	
	N_RANK = 50
	N_CANDIDATE = 100
	recall = {}
	for i in range(N_RANK):
		recall["@{}".format(i)] = 0.0

	predictions = np.argsort(-probs_aggre, axis=1)

	# compute recall
	for rank in range(N_RANK):
		correct_prediction = np.equal(predictions[:, rank], labels_aggre).astype(float)
		recall['@{}'.format(rank+1)] = sum(correct_prediction)

	for rank in range(N_RANK):
		recall['@{}'.format(rank+1)] += recall['@{}'.format(rank)]

	num_examples = labels_aggre.shape[0]
	for rank in range(N_RANK):
		recall['@{}'.format(rank+1)] = recall['@{}'.format(rank+1)] / num_examples
	
	# compute mrr
	mrr = 0.0
	for example in range(len(labels_aggre)):
		for candidate in range(N_CANDIDATE):
			if predictions[example][candidate] == labels_aggre[example]:
				mrr += 1.0 / (candidate+1)
			# else:
				# print("wrong")
				
	mrr = mrr / len(labels_aggre)

	return recall, mrr
