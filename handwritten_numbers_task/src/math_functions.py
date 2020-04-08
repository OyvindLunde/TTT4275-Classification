def find_distance(m1, m2):
    dist = 0
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            dist += (m1[i][j]-m2[i][j])**2
    
    return dist

m1 = [[10,2],[3,4]]
m2 = [[1,2],[3,4]]

print(find_distance(m1,m2))