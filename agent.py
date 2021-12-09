import tensorflow as tf
from policy_network import PolicyNetwork
from doc_gan import DocGan
from seq_gan import SeqGan
from pre_process import get_batch_with_test
import tensorflow_probability as tfp
import numpy as np

class Agent:
    def __init__(self, feature_dim, lr, reward_decay):
        self.feature_dim = feature_dim
        self.lr = lr
        self.gamma = reward_decay
        self.policy = PolicyNetwork()
        self.policy.compile(tf.optimizers.Adam(learning_rate=0.0003))
        self.gan = DocGan()
        self.gan.compile(tf.optimizers.Adam(learning_rate=0.0003))
        self.gan2 = SeqGan()
        self.gan2.compile(tf.optimizers.Adam(learning_rate=0.0003))
        self.ep_docs = []
        self.ep_label = []

    def calcu_DCG(self, labels):
        DCG = []
        for i in range(len(labels)):
            if i == 0:
                DCG.append(np.power(2.0, labels[i]) - 1.0)
            else:
                DCG.append((np.power(2.0, labels[i]) - 1.0) / np.log2(i + 1))
        return DCG

    def calcu_reward(self, labels):
        reward_sum = 0
        DCG = self.calcu_DCG(labels)
        for t in range(len(labels)):
            reward_sum += (0.99 ** t) * DCG[t]
        return reward_sum

    def choose_doc(self, remain_doc_feature, training=True):
        if training == True:
            scores = self.policy(remain_doc_feature)
            scores = tf.reshape(scores, [scores.shape[0]])
            prob = tf.compat.v1.nn.softmax(scores)
            return prob
        else:
            scores = self.policy(remain_doc_feature)
            scores = tf.reshape(scores, [scores.shape[0]])
            d_index = tf.argmax(scores)
            return d_index

    def sorted_docs(self, doc_feature, doc_label):
        sorted_doc = []
        sorted_indices = np.argsort(-doc_label)
        for num in sorted_indices:
            sorted_doc.append(doc_feature[num])
        return sorted_doc, sorted_indices[0]

    def train_state_gan(self, n_batch, n_sample, number, sorted_doc, sorted_label, doc_feature, doc_label):
        with tf.GradientTape(persistent=True) as tape:
            loss_sum = 0
            for t in range(n_batch):
                pos_docs = []
                fake_docs = []
                pos_labels = []
                fake_labels = []
                sorted_feature = np.array(sorted_doc[:number])
                sorted_label = np.array(sorted_label[:number], dtype="float32")
                probs2 = tf.compat.v1.nn.softmax(sorted_label)
                probs1 = self.choose_doc(doc_feature, training=True)
                action_set1 = tfp.distributions.Categorical(probs=probs1).sample(n_sample).numpy()
                action_set2 = tfp.distributions.Categorical(probs=probs2).sample(n_sample).numpy()
                for j in range(n_sample):
                    fake_docs.append(doc_feature[action_set1[j]])
                    fake_labels.append(doc_label[action_set1[j]])
                    pos_docs.append(sorted_feature[action_set2[j]])
                    pos_labels.append(sorted_label[action_set2[j]])
                pos_labels = np.array(pos_labels)
                pos_docs, _ = self.sorted_docs(pos_docs, pos_labels)
                fake_labels = np.array(fake_labels)
                fake_docs, _ = self.sorted_docs(fake_docs, fake_labels)
                fake_docs = np.reshape(np.array(fake_docs), (1, n_sample, 45))
                p_fake = tf.sigmoid(self.gan2(fake_docs))
                pos_docs = np.reshape(np.array(pos_docs), (1, n_sample, 45))
                p_true = tf.sigmoid(self.gan2(pos_docs))
                loss = -(tf.compat.v1.log(p_true) + tf.compat.v1.log(1 - p_fake))
                loss_sum += loss
            loss_sum /= n_batch
        trainable_variables = self.gan2.trainable_variables
        gradient = tape.gradient(loss_sum, trainable_variables)
        self.gan2.optimizer.apply_gradients(zip(gradient, trainable_variables))

    def train_doc_gan(self, pos_num, sorted_doc, sorted_label, doc_feature):
        with tf.GradientTape(persistent=True) as tape:
            sorted_doc = np.array(sorted_doc[:pos_num])
            sorted_label = np.array(sorted_label[:pos_num], dtype="float32")
            true_probs = tf.compat.v1.nn.softmax(sorted_label)
            fake_docs = []
            pos_docs = []
            probs = self.choose_doc(doc_feature, training=True)
            action_set1 = tfp.distributions.Categorical(probs=probs).sample(pos_num).numpy()
            action_set0 = tfp.distributions.Categorical(probs=true_probs).sample(pos_num).numpy()
            for j in range(pos_num):
                fake_docs.append(doc_feature[action_set1[j]])
                pos_docs.append(sorted_doc[action_set0[j]])
            fake_docs = np.array(fake_docs)
            f = tf.sigmoid(self.gan(fake_docs))
            pos_docs = np.array(sorted_doc)
            p = tf.sigmoid(self.gan(pos_docs))
            loss = -tf.reduce_mean(tf.compat.v1.log(p)+tf.compat.v1.log(1-f))
        trainable_variables = self.gan.trainable_variables
        gradient = tape.gradient(loss, trainable_variables)
        self.gan.optimizer.apply_gradients(zip(gradient, trainable_variables))

    def train_policy(self, pos_num, n_sample, doc_feature, doc_label):
        for step in range(pos_num):
            docs = []
            labels = []
            with tf.GradientTape(persistent=True) as tape:
                probs = self.choose_doc(doc_feature, training=True)
                action_set = tfp.distributions.Categorical(probs=probs).sample(n_sample).numpy()
                for j in range(n_sample):
                    docs.append(doc_feature[action_set[j]])
                    labels.append(doc_label[action_set[j]])

                index = action_set[0]
                docs = tf.reshape(np.array(docs), [1, n_sample, 45])
                state_reward = tf.sigmoid(self.gan2(docs))
                doc = doc_feature[index]
                label = doc_label[index]
                log_prob = tf.compat.v1.log(probs[index])
                doc = np.reshape(np.array(doc), (1, 45))
                gan_reward = 2*(tf.sigmoid(self.gan(doc))-0.5)
                if doc_label[index] == 0:
                    state_reward = doc_label[index]*state_reward
                reward = 2 ** label - 1

                loss = - log_prob * (reward + gan_reward + state_reward)
                self.store(doc_feature[index], doc_label[index])
                doc_feature, doc_label = np.delete(doc_feature, index, 0), np.delete(doc_label, index, 0)
            trainable_variables = self.policy.trainable_variables
            gradient = tape.gradient(loss, trainable_variables)
            self.policy.optimizer.apply_gradients(zip(gradient, trainable_variables))

    def predict(self, dataset):
        reward_sum = 0
        label_collection = []
        for data in get_batch_with_test(dataset, 45):
            doc_feature = data[0]
            doc_label = data[1]
            doc_len = data[2]
            for step in range(doc_len):
                index = self.choose_doc(doc_feature, training=False)
                current_doc = doc_feature[index]
                current_label = doc_label[index]
                doc_feature, doc_label = np.delete(doc_feature, index, 0), np.delete(doc_label, index, 0)
                self.store(current_doc, current_label)
            reward = self.calcu_reward(self.ep_label)
            label_collection.append(self.ep_label)
            self.reset()
            reward_sum += reward
        return label_collection, reward_sum
                
    def reset(self):
        self.ep_label = []
        self.ep_docs = []

    def store(self, doc_feature, label):
        self.ep_docs.append(doc_feature)
        self.ep_label.append(label)

    def save_models(self):
        print('...saving models...')
        self.policy.save_weights(self.policy.checkpoint_file)
        self.gan.save_weights(self.gan.checkpoint_file)

    def load_model(self):
        print('...loading model...')
        self.policy.load_weights(self.policy.checkpoint_file)
        self.gan.load_weights(self.gan.checkpoint_file)



