import numpy as np

def Cloud(p, nvec):
    # Variable initialization
    m    = len(p[:,0])                                                              # The size if the triangulation is obtained.
    vec  = np.zeros([m, nvec], dtype=int) - 1                                       # The array for the neighbors is initialized.
    dmin = np.zeros([m, 1]) + 1                                                     # dmin initialization with a "big" value.

    # Delta computation for finding neighbors
    for i in np.arange(m):                                                          # For each of the nodes.
        x    = p[i,0]                                                               # x coordinate of the central node.
        y    = p[i,1]                                                               # y coordinate of the central node.
        for j in np.arange(m):                                                      # For all the nodes.
            if i != j:                                                              # If the the node is different to the central one.
                x1 = p[j,0]                                                         # x coordinate of the possible neighbor.
                y1 = p[j,1]                                                         # y coordinate of the possible neighbor.
                d  = np.sqrt((x - x1)**2 + (y - y1)**2)                             # Distance from the possible neighbor to the central node.
                dmin[i] = min(dmin[i],d)                                            # Look for the minimum distance.
    dist = (3/2)*max(max(dmin))

    # Search of the neighbor nodes
    for i in np.arange(m):                                                          # For each of the nodes.
        x    = p[i,0]                                                               # x coordinate of the central node.
        y    = p[i,1]                                                               # y coordinate of the central node.
        temp = 0                                                                    # Temporal variable as a counter.
        for j in np.arange(m):                                                      # For all the interior nodes.
            if i != j:                                                              # Check that we are not working with the central node.
                x1 = p[j,0]                                                         # x coordinate of the possible neighbor.
                y1 = p[j,1]                                                         # y coordinate of the possible neighbor.
                d  = np.sqrt((x - x1)**2 + (y - y1)**2)                             # Distance from the possible neighbor to the central node.
                if d < dist:                                                        # If the distance is smaller or equal to the tolerance distance.
                    if temp < nvec:                                                 # If the number of neighbors is smaller than nvec.
                        vec[i,temp] = j                                             # Save the neighbor.
                        temp       += 1                                             # Increase the counter by 1.
                    else:                                                           # If the number of neighbors is greater than nvec.
                        x2 = p[vec[i,:],0]                                          # x coordinates of the current neighbor nodes.
                        y2 = p[vec[i,:],1]                                          # y coordinates of the current neighbor nodes.
                        d2 = np.sqrt((x - x2)**2 + (y - y2)**2)                     # The total distance from all the neighbors to the central node.
                        I  = np.argmax(d2)                                          # Look for the greatest distance.
                        if d < d2[I]:                                               # If the new node is closer than the farthest neighbor.
                            vec[i,I] = j                                            # The new neighbor replace the farthest one.
    return vec