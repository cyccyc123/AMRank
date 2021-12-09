import numpy as np
import os
from pre_process import get_batch
from agent import Agent
from evaluation import evaluation_ranklists
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def calcu_num(doc_feature, doc_label):
    pos_num = 0
    max_num = 0
    for t in range(len(doc_feature)):
        if doc_label[t] != 0:
            pos_num += 1
        if doc_label[t] == 2:
            max_num += 1
    return max_num, pos_num

def sorted_docs(doc_feature, doc_label):
    sorted_doc = []
    sorted_label = []
    sorted_indices = np.argsort(-doc_label)
    for num in sorted_indices:
        sorted_doc.append(doc_feature[num])
        sorted_label.append(doc_label[num])
    return sorted_doc, sorted_label

def train():
    file_name = "Fold5"
    train_path = "../dataset/OHSUMED/"+file_name+"/trainingset.txt"
    test_path = "../dataset/OHSUMED/"+file_name+"/testset.txt"
    QRL_L2R = Agent(feature_dim=45, lr=0.001, reward_decay=0.99)
    n_sample = 32
    n_batch = 4
    # QRL_L2R.load_model()
    for i in range(1000):
        print("\nepoch"+str(i+1)+"\n")
        for data in get_batch(train_path, 45):
            doc_feature = data[0]
            doc_label = data[1]
            doc_len = data[2]
            qid = data[3]
            max_num, pos_num = calcu_num(doc_feature, doc_label)
            if pos_num == 0:
                continue
            sorted_doc, sorted_label = sorted_docs(doc_feature, doc_label)
            QRL_L2R.train_doc_gan(pos_num, sorted_doc, sorted_label, doc_feature)
            QRL_L2R.train_state_gan(n_batch, n_sample, pos_num, sorted_doc, sorted_label, doc_feature, doc_label)
            QRL_L2R.train_policy(pos_num, n_sample, doc_feature, doc_label)
            reward = QRL_L2R.calcu_reward(QRL_L2R.ep_label)
            QRL_L2R.reset()
            print("training, qid :{} with_length : {}, reward : {}".format(qid, doc_len, reward))

        # test evaluation
        test_predict_label_collection, test_reward = QRL_L2R.predict(test_path)
        test_MAP, test_NDCG_at_1, test_NDCG_at_3, test_NDCG_at_5, test_NDCG_at_10, test_NDCG_at_20, test_MRR, test_P = evaluation_ranklists(
            test_predict_label_collection)
        test_result_line = "## test_MAP : {}, test_NDCG_at_1 : {}, test_NDCG_at_3 : {}, test_NDCG_at_5 : {}, test_NDCG_at_10 : {}, test_NDCG_at_20 : {}, test_MRR@20 : {}, test_P@20 : {}, \ntest_reward : {}".format(
            test_MAP, test_NDCG_at_1, test_NDCG_at_3, test_NDCG_at_5, test_NDCG_at_10, test_NDCG_at_20, test_MRR,
            test_P, test_reward)
        print(test_result_line)

        #save param
        QRL_L2R.save_models()

if __name__ == "__main__":
    train()