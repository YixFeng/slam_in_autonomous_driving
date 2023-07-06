//
// Created by yixfeng on 6/07/23.
//

#ifndef SLAM_IN_AUTO_DRIVING_DEGENERATION_DETECTION_H
#define SLAM_IN_AUTO_DRIVING_DEGENERATION_DETECTION_H

#include "ch6/dbscan.h"
#include "common/math_utils.h"

namespace sad {

class DegenDetection {
   public:
    using Point2d = pcl::PointXY;
    using Cloud2d = pcl::PointCloud<Point2d>;

    DegenDetection() {}

    void SetScan(Scan2d::Ptr scan) {
        scan_ = scan;
        BuildKdTree();
    }

    int GetDegenCount() const {
        return degen_count;
    }

    void AddCount() {
        ++degen_count;
    }

    /**
     * 初始化DBSCAN聚类
     * @param coreMinPts 核心点邻域内最少点数
     * @param minPtsPerCluster 每个cluster最少点数
     * @param maxPtsPerCluster 每个cluster最多点数
     * @param eps 邻域距离
     */
    void InitDBSCAN(int coreMinPts, int minPtsPerCluster, int maxPtsPerCluster, double eps);

    bool DetectDegeneration();

    void ShowClustering(cv::Mat& image, int image_size, float resolution);

    void BuildKdTree();

   private:
    Scan2d::Ptr scan_ = nullptr;
    Cloud2d::Ptr cloud_;
    pcl::search::KdTree<Point2d>::Ptr kdtree_;

    DBSCANSimpleCluster<Point2d> dbscan_;
    std::vector<pcl::PointIndices> cluster_indices;
    std::vector<Vec2d> effective_pts;
    std::vector<Vec3d> line_coeffs;
    std::vector<double> slope; // 存储斜率
    double slope_tolerance = 0.1;
    int degen_count = 0;
};

}

#endif  // SLAM_IN_AUTO_DRIVING_DEGENERATION_DETECTION_H
