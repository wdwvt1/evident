#!/usr/bin/env python
# File created on 16 Oct 2012
from __future__ import division

__author__ = "William Van Treuren"
__copyright__ = "Copyright 2012, Evident"
__credits__ = ["William Van Treuren"]
__license__ = "GPL"
__version__ = ".9dev"
__maintainer__ = "William Van Treuren"
__email__ = "wdwvt1@gmail.com"
__status__ = "Development"

from collections import defaultdict
from itertools import combinations
from numpy import array, zeros
from qiime.rarefaction import get_rare_data
from qiime.beta_diversity import single_object_beta
from qiime.principal_coordinates import pcoa
from pcoa import get_pcoa_ellipsoid_coords

"""This library is used to calculate the percentage of the time that, when 
rarifying at a given depth, the pcoa-space treatment clusters overlap."""



def calc_dist_in_3d(p1,p2):
    """calculate the distance between p1 and p2 in R3."""
    return ((array(p1)-array(p2))**2).sum()**.5

def calculate_enclosing_sphere(points):
    """calcs center and radius of a sphere that encloses everything in points.
    Points is a nx3 array with col0=x,col1=y,col2=z."""
    center = points.sum(0)/points.shape[0]
    rs = [calc_dist_in_3d(center, p) for p in points]
    return center, max(rs)

def test_sphere_intersection(center1, r1, center2, r2):
    """tests if two R3 spheres have any intersecting points.
    r1+r2 > sqrt[(x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2)] because co-linear radii
    intersecting is necessary and sufficient for sphere intersection. 
    """
    # original code -- might be faster, leave for future possible use
    #d = reduce(lambda x,y: x+y,[(center1[i] - center2[i])**2 for i in [0,1,2]])
    #return d**.5 < r1+r2
    return calc_dist_in_3d(center1, center2) < r1+r2

def get_category_clusters(category, pcoa_centers_radii, parsed_mf):
    """finds the groups of sampleids that have the same value for 'category'.
    Notes:
     Returns the different values of the categories, i.e. the treatments, and
     the dict which maps category values to sampleids that have that value.
    """
    category_to_sid = defaultdict(list)
    for k in pcoa_centers_radii.keys():
        category_to_sid[parsed_mf[k][category]].append(k)
    return category_to_sid.keys(), category_to_sid

def get_locations_in_r3(sampleids, pcoa_centers_radii):
    """finds the x,y,z=pc1,pc2,pc3 pts in pcoa space for the given sampleids.
    Notes:
     Depends on the fact that pcoa_centers_radii has an entry for each sample
     called 'center'."""
    flat_vals = array([pcoa_centers_radii[i]['center'][:3] for i in sampleids])
    return flat_vals.reshape(len(sampleids),3) #wont fit on one line at 80 chars

def cluster_intersections(category, parsed_mf, pcoa_centers_radii,
    intersections=None):
    """Calcs what treatments overlap in pcoa space from pcoa_centers_radii
    Notes:
     Taking the given category and the given pcoa_centers_radii, this function
     calculates which treatments overlap in pcoa space. Each treatment has 
     some number of points corresponding to the samples that have that 
     treatment. This function calculates the sphere in r3 that would contain
     all the points for a given treatment and then tests if that sphere 
     intersects the spheres of other treatments.
    """
    # treatments are values of category, cts is category_to_sampleid
    treatments, cts = get_category_clusters(category,pcoa_centers_radii,
        parsed_mf)
    spheres = combinations(treatments, 2)
    # intersections will store locations where intersections occur
    if intersections is None:
        intersections = zeros((len(treatments),len(treatments)))
    for s1,s2 in spheres: #s1,s2 are strs, names of treatments
        s1_pts = get_locations_in_r3(cts[s1], pcoa_centers_radii)
        s2_pts = get_locations_in_r3(cts[s2], pcoa_centers_radii)
        s1_center, s1_radius = calculate_enclosing_sphere(s1_pts)
        s2_center, s2_radius = calculate_enclosing_sphere(s2_pts)
        if test_sphere_intersection(s1_center, s1_radius, s2_center, s2_radius):
            # the spheres do intersect
            intersections[treatments.index(s1)][treatments.index(s2)]+=1
    return intersections, treatments



def calculate_cluster_intersections(category, parsed_mf, biom_object,
    metric, num_seqs, iterations, axes=3, tree_object=None):
    """Calculate number of times clusters intersect.
    Notes:
     This function is the main function of the library and utilizes all the
     other functions. The general workflow is take the passed category,
     calculate the pcoa coordinates of each treatment in that category, find the
     sphere which encloses all those pcoa coordinates, and then calculate the 
     intersections between those spheres.
    Inputs:
     category - str, the category in the mapping file by which points should be 
     compared, e.g., 'AGE' or 'SEX'.
     parsed_mf - dict, the mapping file for the given study.
     biom_object - biom, the biom file for the given study.
     metric - str, metric in pycogent that will be used to calculate the
     distances between samples.
     num_seqs - int, the number of sequences to rarify the biom table at. 
     iterations - int, the number of independent rarifaction, distance 
     calculation, intersection calculation cycles to go through.
     axes - int, the number of axes to pull out from the pcoa_centers_radii. 
     currently not utilized.
     tree_object - pycogent tree_object, required if the metric specified is a
     phylogenetic distance metric.
    Outputs:
         A  B  C
        -- -- --
     A |0  1  5
     B |0  0  7
     C |0  0  0

     Where A,B,C are the treatments in the passed category, and the values are
     the number of times the clusters for treatment i and j overlapped.
     An additional output is 'treatments' which is just the list [A,B,C] in the 
     order of the marginals of the matrix.
    """
    # set intersections to none so cluster_intersections knows to create it
    intersections = None
    for i in range(iterations):
        rarefied_bt = get_rare_data(biom_object, num_seqs)
        beta_div_dm = single_object_beta(rarefied_bt, metric, tree_object)
        pcoa_pts = pcoa(beta_div_dm)
        # tmp is a dict: {Sample:[5,6,7]}, not needed here
        pcoa_centers_radii, tmp = \
            get_pcoa_ellipsoid_coords([pcoa_pts],axes, parsed_mf.keys())
        intersections, treatments = cluster_intersections(category, parsed_mf,
            pcoa_centers_radii, intersections)
    return intersections, treatments











