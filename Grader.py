python

class HomeworkGrader:
    def __init__(self, yolo_model_path):
        self.yolo_model_path = yolo_model_path

    def _detect_answers(self, image):
        results = [("right", 0.9, (50, 50, 150, 150)),
                   ("false", 0.8, (200, 200, 300, 300)),
                   ("half", 0.75, (350, 350, 450, 450))]
        return results

    def _calculate_score(self, detections):
        total_count = len(detections)
        right_count = sum(1 for d in detections if d[0] == "right")
        half_count = sum(1 for d in detections if d[0] == "half")

        if total_count == 0:
            return 0

        per_question_score = 100.0 / total_count
        score = right_count * per_question_score + half_count * per_question_score * 0.5
        return score

    def grade_homework(self, image_path):
        image = cv2.imread(image_path)
        detections = self._detect_answers(image)
        score = self._calculate_score(detections)


   ......
