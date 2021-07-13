import csv
import sys
import calendar

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    data_evidence_result = []
    data_labels_result = []

    with open(filename) as file:
        csv_data = csv.reader(file)
        next(csv_data)

        months = {name:num for num, name in enumerate(calendar.month_abbr) if num}

        for line in csv_data:
            total_cols = len(line)
            formatted_fields = []

            for field_key, field in enumerate(line[:total_cols-1]):
                formatted_fields.append(convert_data(field, field_key, months))

            data_evidence_result.append(formatted_fields)
            data_labels_result.append(1 if line[total_cols-1] == "TRUE" else 0)

    return (data_evidence_result, data_labels_result)

def convert_data(field, key, months):
    integer_converter = { "TRUE":1, "FALSE":0 }
    visitor_converter = { "Returning_Visitor":1, "New_Visitor":0, "Other":0 }

    to_int_id_list = [0, 2, 4, 11, 12, 13, 14]
    to_float_id_list = [1, 3, 5, 6, 7, 8, 9]
    bool_to_float_id_list = [16]
    visitor_id_list = [15]
    month_id_list = [10]

    if key in to_int_id_list:
        return int(field)
    elif key in to_float_id_list:
        return float(field)
    elif key in bool_to_float_id_list:
        return integer_converter[field]
    elif key in visitor_id_list:
        return visitor_converter[field]
    elif key in month_id_list:
        if field == 'June':
            field = 'Jun'

        return months[field]-1
    else:
        return field


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neighbors = KNeighborsClassifier(n_neighbors=1)
    neighbors.fit(evidence, labels)

    return neighbors


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    correct = 0
    incorrect = 0
    total = 0
    for label, prediction in zip(labels, predictions):
        total += 1
        if label == prediction:
            correct += 1
        else:
            incorrect += 1

    return ((incorrect/total), (correct/total))

if __name__ == "__main__":
    main()
