from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from typing import List, Union


class SelfTokenizer:
    def __init__(self, vocab_file=None):
        """
        :param vocab_file: 关于vocab的json文件。
        """
        self.vocab_file = vocab_file
        if self.vocab_file is not None:
            self.vocab = self.read_json()

    def read_json(self):
        with open(self.vocab_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def train(self, files: Union[str, List[str]]=None, special_tokens: Union[str, List[str]]=None):
        """
        :param files: [file1, file2, file3.....]
        :param special_tokens: ["__null__", "__start__", "__end__", "__unk__", "__newln__"]， 也可以自己指定。
        """
        assert files is not None, "files cannot be None"
        tokenizer = Tokenizer()
        datas = []
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                data = f.readlines()
                new_data = [" ".join([n for n in i]) for i in data]
                datas.extend(new_data)
        tokenizer.fit_on_texts(datas)
        vocab_dir = tokenizer.word_index
        if special_tokens is None:
            vocab = {}
        else:
            vocab = {special_tokens[i]:i for i in range(len(special_tokens))}
        for k in vocab_dir.keys():
            vocab[k] = len(vocab)
        self.vocab_dir = vocab
        return len(vocab)

    def save(self, file):
        """
        :param file: file or file.json
        """
        if ".json" not in file:
            file += ".json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump(self.vocab_dir, f, ensure_ascii=False, indent=4)

    def decoder(self, sentence_num, remove_flag=True):
        """
        :param sentence_num: tensorflow的张量数组    维度为2
        :param remove_flag:去除特殊令牌
        """
        vocab = {v:k for k, v in self.vocab.items()}
        sentences = []
        axis = tf.rank(sentence_num).numpy()
        if axis == 1:
            sentence_num = sentence_num[None, :]
        for sentence in sentence_num:
            if remove_flag:
                new_sentence = [vocab[i] for i in sentence.numpy() if i not in [0, 1, 2, 3, 4]]
            else:
                new_sentence = [vocab[i] for i in sentence.numpy()]
            sentences.append("".join(new_sentence))
        return sentences

    def encoder(self, text, add_special_tokens=False, padding=False, truncation=False, max_len=None):
        """
        :param text: [sentence1, sentence2, sentence3.....]
        :param add_special_tokens: 添加开始标志和结束标志。
        :param padding:长度统一
        :param truncation:和padding一样，一起为True
        :param max_len:最大长度
        :param beginning_to_end:剪取的顺序，True为从index为零开始到max_len，False为从index为-1开始到-max_len。
        """
        vocab = self.vocab
        text_num = []
        for t in text:
            new_t = [n for n in t]
            num_ls = []
            for i in new_t:
                if i == "" or i == " ":
                    continue
                try:
                    num = vocab[i]
                except:
                    num = vocab["__unk__"]
                num_ls.append(num)
            text_num.append(num_ls)

        if padding and truncation and max_len and not add_special_tokens:
            text_num = pad_sequences(text_num, padding="post", truncating="post", maxlen=max_len)

        elif padding and truncation:
            if add_special_tokens and not max_len:
                new_text = []
                for i in text_num:
                    i.append(vocab["__end__"])
                    i.insert(0, vocab["__start__"])
                    new_text.append(i)
                text_num = new_text
                text_num = pad_sequences(text_num, padding="post", truncating="post")

            elif add_special_tokens and max_len:
                new_text = []
                for i in text_num:
                    if len(i) < max_len - 2:
                        i.append(vocab["__end__"])
                        i.insert(0, vocab["__start__"])
                    else:
                        i = i[:max_len-2]
                        i.append(vocab["__end__"])
                        i.insert(0, vocab["__start__"])
                    new_text.append(i)
                text_num = new_text
                text_num = pad_sequences(text_num, padding="post", truncating="post", maxlen=max_len)

            elif not add_special_tokens and not max_len:
                new_text = []
                for i in text_num:
                    new_text.append(i)
                text_num = new_text
                text_num = pad_sequences(text_num, padding="post", truncating="post")

            elif not add_special_tokens and max_len:
                new_text = []
                for i in text_num:
                    new_text.append(i)
                text_num = new_text
                text_num = pad_sequences(text_num, padding="post", truncating="post", maxlen=max_len)

        else:
            assert len(text_num) == 1, "You should specify padding=True and truncating=True"

            if max_len:
                text_num = [text_num[0][:max_len]]

            if add_special_tokens:
                new_text = text_num[0]
                new_text.append(vocab["__end__"])
                new_text.insert(0, vocab["__start__"])
                text_num = [new_text]
        input_ids = tf.constant(text_num)
        input_act = tf.constant(input_ids.numpy()>0, dtype="int32")
        return {
            "input_ids":input_ids,
            "attention_mask":input_act
        }


if __name__ == '__main__':
    tokenizer = SelfTokenizer("vocab.json")
    inputs = tokenizer.encoder(["你好，我 是 唐小书。 | 好的啊。", "好的。"],padding=True, truncation=True, add_special_tokens=True, max_len=20)
    input_ids = inputs["input_ids"]
    print(input_ids)
    print(tokenizer.decoder(input_ids))
