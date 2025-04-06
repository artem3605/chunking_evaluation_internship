import math


class Evaluation:

    def __init__(self, corpus, questions, chunker):
        self.corpus = corpus
        self.questions = questions
        self.chunker = chunker

    def calc_query_metrics(self, question_id, retrieved_chunks):
        """
        Calculate iou, precision and recall for the given question_id and retrieved chunks.
        """
        total_length_of_retrieved_chunks = 0
        retrieved_ixs = set()
        for chunk in retrieved_chunks:
            total_length_of_retrieved_chunks += len(chunk)
            start_index = self.corpus.index(chunk)

            end_index = start_index + len(chunk)
            retrieved_ixs.update(set(range(start_index, end_index)))

        golden_ixs = set()
        for chunk in self.questions.loc[question_id]["references"]:
            start_index = chunk["start_index"]
            end_index = chunk["end_index"]
            golden_ixs.update(set(range(start_index, end_index)))

        iou = len(retrieved_ixs.intersection(golden_ixs)) / (
                len(golden_ixs) + total_length_of_retrieved_chunks - len(retrieved_ixs.intersection(golden_ixs)))
        precision = len(retrieved_ixs.intersection(golden_ixs)) / total_length_of_retrieved_chunks
        recall = len(retrieved_ixs.intersection(golden_ixs)) / len(golden_ixs)
        return iou, precision, recall

    def calc_metrics(self, retrieved_chunks):
        """
        Calculate mean iou, precision and recall for all questions.
        """
        iou_list = []
        precision_list = []
        recall_list = []
        for question_id in self.questions.index:
            iou, precision, recall = self.calc_query_metrics(question_id, retrieved_chunks[question_id])
            iou_list.append(iou)
            precision_list.append(precision)
            recall_list.append(recall)

        mean_iou = sum(iou_list) / len(iou_list)
        mean_precision = sum(precision_list) / len(precision_list)
        mean_recall = sum(recall_list) / len(recall_list)

        std_iou = math.sqrt(sum((x - mean_iou) ** 2 for x in iou_list) / len(iou_list))
        std_precision = math.sqrt(sum((x - mean_precision) ** 2 for x in precision_list) / len(precision_list))
        std_recall = math.sqrt(sum((x - mean_recall) ** 2 for x in recall_list) / len(recall_list))

        return {
            "mean_iou": mean_iou,
            "std_iou": std_iou,
            "mean_precision": mean_precision,
            "std_precision": std_precision,
            "mean_recall": mean_recall,
            "std_recall": std_recall
        }
