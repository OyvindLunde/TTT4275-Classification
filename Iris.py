import extract_data as ed
import math_functions as mf

def main():
    x_train, x_test, y_train, y_test, t = ed.get_sets()
    W = mf.calculate_W(1000,0.00001, t, x_train)
    print(ed.confusion_matrix())

main()