import extract_data as ed
import math_functions as mf

def main():
    x_train, x_test, y_train, y_test, t = ed.get_sets()
    #print(x_train)
    W = mf.calculate_W(1000,0.0001, t, x_train)
    print(W)

main()