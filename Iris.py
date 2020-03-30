import extract_data as ed
import math_functions as mf

def main():
    x_train, x_test, y_train, y_test, t = ed.get_sets()
    W = mf.calculate_W(100,0.00001, t, x_train)
    #print(W)
    pred = mf.predict(x_train, W, t)
    #print(pred)
main()