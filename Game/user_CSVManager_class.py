import csv

class UserCSVManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def add_user(self, user_name):
        users = self.get_users()
        if user_name not in users:
            with open(self.file_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([user_name, 0, 0, 0])

    def get_users(self):
        users = []
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                if row:  # Check if the row is not empty
                    users.append(row[0])
        return users

    def get_user_scores(self, user_name):
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header row
            for row in reader:
                if row:
                    if row[0] == user_name:
                        return {
                            'user_name': row[0],
                            'Max_score_level_1': int(row[1]),
                            'Max_score_level_2': int(row[2]),
                            'Max_score_level_3': int(row[3])
                        }

    def update_high_score(self, user_name, scores):
        data = []
        header = None
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Get the header row
            for row in reader:
                if(len(row)==4):
                    if row[0] == user_name:
                        for i in range(1, 4):
                            if int(scores[i]) > int(row[i]):
                                row[i] = str(scores[i])
                    data.append(row)
        with open(self.file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)  # Write the header row
            writer.writerows(data)

    def get_top_users(self):
        """ 
        hieghst score for each level
        return : {
            'Max_score_level_1': [
                                ('User1', 100),
                                ('User2', 95),
                                ('User3', 90),....
                }

            Max_score_level_2':[ 'User', score(int) ] ....

        """
        top_users = {}
        with open(self.file_path, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)  # header row
            score_columns = header[1:]  #   column
            for column in score_columns:
                top_users[column] = []
            for row in reader:  # each row in file
                if row:
                    user_name = row[0]
                    scores = [int(score) for score in row[1:]]
                    for i, score in enumerate(scores):
                        if len(top_users[score_columns[i]]) < 10:
                            
                            top_users[score_columns[i]].append((user_name, score))
                        else:
                            min_score = min(top_users[score_columns[i]], key=lambda x: x[1])
                            if score > min_score[1]:
                                top_users[score_columns[i]].remove(min_score)
                                top_users[score_columns[i]].append((user_name, score))
                                top_users[score_columns[i]] = sorted(top_users[score_columns[i]], key=lambda x: x[1], reverse=True)
        return top_users
