//
// Created by yixfeng on 19/06/23.
//

#ifndef SLAM_IN_AUTO_DRIVING_NANOFLANN_UTILS_H
#define SLAM_IN_AUTO_DRIVING_NANOFLANN_UTILS_H

#include <nanoflann.hpp>
#include "common/point_cloud_utils.h"
#include "common/point_types.h"
#include "common/sys_utils.h"
#include "common/math_utils.h"

// Copied from nanoflann utils.h
template <typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z;
    };

    using coord_t = T;  //!< The type of each coordinate

    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

using nano_flann_tree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>,
                        PointCloud<float>, 3, size_t>;

bool GetClosestPointNanoFlann(nano_flann_tree &tree, const sad::CloudPtr &query, std::vector<std::pair<size_t, size_t>> &matches, size_t k) {
    matches.resize(query->size() * k);

    std::vector<size_t> index(query->size());
    for (size_t i = 0; i < query->points.size(); i++) {
        index[i] = i;
    }

    std::for_each(std::execution::seq, index.begin(), index.end(), [&tree, &query, &matches, &k](size_t idx) {
        size_t num_results = k;
        std::vector<size_t> ret_index(num_results);
        std::vector<float> out_dist_sqr(num_results);
        auto pt = query->points[idx];
        float query_pt[3] = {pt.x, pt.y, pt.z};
        num_results = tree.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
        for (size_t i = 0; i < k; i++) {
            matches[idx * k + i].second = idx;
            if (i < num_results) {
                matches[idx * k + i].first = ret_index[i];
            } else {
                matches[idx * k + i].first = sad::math::kINVALID_ID;
            }
        }
    });

    return true;
}

bool GetClosestPointNanoFlannMT(nano_flann_tree &tree, const sad::CloudPtr &query, std::vector<std::pair<size_t, size_t>> &matches, size_t k) {
    matches.resize(query->size() * k);

    std::vector<size_t> index(query->size());
    for (size_t i = 0; i < query->points.size(); i++) {
        index[i] = i;
    }

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&tree, &query, &matches, &k](size_t idx) {
        size_t num_results = k;
        std::vector<size_t> ret_index(num_results);
        std::vector<float> out_dist_sqr(num_results);
        auto pt = query->points[idx];
        float query_pt[3] = {pt.x, pt.y, pt.z};
        num_results = tree.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
        for (size_t i = 0; i < k; i++) {
            matches[idx * k + i].second = idx;
            if (i < num_results) {
                matches[idx * k + i].first = ret_index[i];
            } else {
                matches[idx * k + i].first = sad::math::kINVALID_ID;
            }
        }
    });

    return true;
}

#endif  // SLAM_IN_AUTO_DRIVING_NANOFLANN_UTILS_H
