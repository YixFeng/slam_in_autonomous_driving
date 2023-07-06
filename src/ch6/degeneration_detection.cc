//
// Created by yixfeng on 6/07/23.
//

#include "ch6/degeneration_detection.h"

#include <glog/logging.h>

namespace sad {

void DegenDetection::InitDBSCAN(int coreMinPts, int minPtsPerCluster, int maxPtsPerCluster, double eps) {
    dbscan_.setCorePointMinPts(coreMinPts);
    dbscan_.setClusterTolerance(eps);
    dbscan_.setMinClusterSize(minPtsPerCluster);
    dbscan_.setMaxClusterSize(maxPtsPerCluster);
    dbscan_.setSearchMethod(kdtree_);
    dbscan_.setInputCloud(cloud_);
    cluster_indices.clear();
    dbscan_.extract(cluster_indices);
//    LOG(INFO) << "cluster个数: " << cluster_indices.size();
}

bool DegenDetection::DetectDegeneration() {
    if (cluster_indices.empty()) {
        LOG(ERROR) << "no enough clusters";
        return false;
    }

    // 简化了检测退化场景的过程，仅检测了走廊场景
    if (cluster_indices.size() == 2) {
        line_coeffs.clear();
        for (size_t i = 0; i < cluster_indices.size(); ++i) {
            effective_pts.clear();
            for (size_t j = 0; j < cluster_indices[i].indices.size(); ++j) {
                Point2d pt = cloud_->at(cluster_indices[i].indices[j]);
                effective_pts.emplace_back(Vec2d(pt.x, pt.y));
            }
            Vec3d coeffs;
            math::FitLine2D(effective_pts, coeffs);
            line_coeffs.emplace_back(coeffs);
            slope.emplace_back(- coeffs[0] / coeffs[1]);
        }

        if (slope[0] - slope[1] < slope_tolerance)
            return true;
    }
    return false;
}

void DegenDetection::ShowClustering(cv::Mat& image, int image_size, float resolution) {
    if (image.data == nullptr) {
        image = cv::Mat(image_size, image_size, CV_8UC3, cv::Vec3b(255, 255, 255));
    }

    for (size_t i = 0; i < cluster_indices.size(); ++i) {
        for (size_t j = 0; j < cluster_indices[i].indices.size(); ++j) {
            Point2d pt = cloud_->points[cluster_indices[i].indices[j]];
            int image_x = int(pt.x * resolution + image_size / 2);
            int image_y = int(pt.y * resolution + image_size / 2);
            if (image_x >= 0 && image_x < image.cols && image_y >= 0 && image_y < image.rows) {
                image.at<cv::Vec3b>(image_y, image_x) = cv::Vec3b(
                    (i+1) * 25 % 256, (i+1) * 75 % 256, (i+1) * 125 % 256);
            }
        }
    }
}

void DegenDetection::BuildKdTree() {
    if (scan_ == nullptr) {
        LOG(ERROR) << "scan is not set";
        return;
    }

    cloud_.reset(new Cloud2d);
    for (size_t i = 0; i < scan_->ranges.size(); ++i) {
        if (scan_->ranges[i] < scan_->range_min || scan_->ranges[i] > scan_->range_max) {
            continue;
        }

        double real_angle = scan_->angle_min + i * scan_->angle_increment;

        Point2d p;
        p.x = scan_->ranges[i] * std::cos(real_angle);
        p.y = scan_->ranges[i] * std::sin(real_angle);
        cloud_->points.push_back(p);
    }

    cloud_->width = cloud_->points.size();
    cloud_->is_dense = false;
    kdtree_.reset(new pcl::search::KdTree<Point2d>);
    kdtree_->setInputCloud(cloud_);
}

}