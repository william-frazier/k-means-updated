# Spring 2021
# BU CS 565
# Project 1
# William Frazier

from random import randint
import pandas as pd


def compute_new_centroid(assigned_points):
    """
    Given a list of points assigned to a centroid, this function will return
    the point which is the average of all the points.
    """
    
    assert type(assigned_points) == list and type(assigned_points[0]) == list, "assigned_points must be a list of lists"
    num_points = len(assigned_points)
    dimensions = len(assigned_points[0])
    assert num_points > 0 and dimensions > 0, "There must be at least 1 data point with at least 1 dimension"
    # Should probably include more checks that dimensions are all the same, etc.
    
    new_centroid = [0] * dimensions # Go ahead and initialize the list
    for dimension in range(dimensions):
        avg = 0
        for point in assigned_points:
            avg += point[dimension]
        avg /= num_points
        new_centroid[dimension] = avg
    return new_centroid


def assign_to_centroid(X,centroids):
    """
    Given a list of points X and a list of centroids, assign the data points
    to their closest centroid. Returns a dictionary where keys are index into
    the centroids list and the values are the data points in X.
    """
    
    assert type(X) == list and type(centroids) == list, "Both X and centroids must be lists"
    assert len(centroids) > 0, "Centroids cannot be empty"
    assert len(X) >= len(centroids), "There must be at least as many data points as centroids"
    assert type(X[0]) == list and type(centroids[0]) == list, "Both X and centroids must be lists of lists"
    # Probably should have an assert checking that each list inside X and centroids are non-empty
    
    # Intitialize the dictionary to have a key for each centroid
    # where the value begins as an empty list
    labels = {new_list: [] for new_list in range(len(centroids))}
    for point in X:
        assigned_centroid = 0
        distance_to_centroid = calc_l2sq_norm(point, centroids[assigned_centroid])
        for centroid in range(len(centroids)):
            new_distance = calc_l2sq_norm(point, centroids[centroid])
            if new_distance < distance_to_centroid:
                assigned_centroid = centroid
                distance_to_centroid = new_distance
        labels[assigned_centroid].append(point)
    return labels


def import_X(filename):
    """
    Reads in a csv file and returns a list of values.
    """
    
    df = pd.read_csv(filename, header=None, encoding = 'utf8')
    return df.values.tolist()


def random_init(X,k):
    """
    Pick k random points as centroids (no overlap).
    """
    
    assert type(k) == int and k > 0, "k must be a positive integer"
    num_points = len(X)
    assert num_points >= k, "There must be at least as many data points as k"
    num_centroids = 1
    # Pick a first centroid so we can set the length of the list
    first_centroid = X[randint(0, num_points-1)]
    centroids = [first_centroid] * k # Much faster to initialize the whole array now
    while num_centroids < k:
        # Pick a random point to be a centroid
        centroid = X[randint(0,num_points-1)] 
        # Only 
        if centroid not in centroids:
            centroids[num_centroids] = centroid
            num_centroids += 1
    return centroids




def calc_l2sq_norm(x,y):
    """
    Calculate the L2-squared distance between x and y.
    """
    
    dimensions = len(x)
    assert dimensions == len(y), "Dimension of X does not equal dimension of Y"
    for value in range(dimensions):
        assert(type(x[value]) in [float, int] and type(y[value]) in [float, int]), "Only real numbers can be used in data points"
        
    distance = 0
    for dimension in range(dimensions):
        distance += (x[dimension] - y[dimension])**2
    return distance