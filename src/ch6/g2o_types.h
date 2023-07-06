//
// Created by xiang on 2022/3/22.
//

#ifndef SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
#define SLAM_IN_AUTO_DRIVING_G2O_TYPES_H

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>

#include <glog/logging.h>
#include <opencv2/core.hpp>

#include "common/eigen_types.h"
#include "common/lidar_utils.h"
#include "common/math_utils.h"

#include <pcl/search/kdtree.h>

namespace sad {

class VertexSE2 : public g2o::BaseVertex<3, SE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void setToOriginImpl() override { _estimate = SE2(); }
    void oplusImpl(const double* update) override {
        _estimate.translation()[0] += update[0];
        _estimate.translation()[1] += update[1];
        _estimate.so2() = _estimate.so2() * SO2::exp(update[2]);
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }
};

class EdgeSE2LikelihoodFiled : public g2o::BaseUnaryEdge<1, double, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2LikelihoodFiled(const cv::Mat& field_image, double range, double angle, float resolution = 10.0)
        : field_image_(field_image), range_(range), angle_(angle), resolution_(resolution) {}

    /// 判定此条边是否在field image外面
    bool IsOutSide() {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2i pf = (pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2)).cast<int>();  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            return false;
        } else {
            return true;
        }
    }

    void computeError() override {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2d pf =
            pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            _error[0] = math::GetPixelValue<float>(field_image_, pf[0], pf[1]);
        } else {
            _error[0] = 0;
            setLevel(1);
        }
    }

    void linearizeOplus() override {
        VertexSE2* v = (VertexSE2*)_vertices[0];
        SE2 pose = v->estimate();
        float theta = pose.so2().log();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Vec2d pf =
            pw * resolution_ + Vec2d(field_image_.rows / 2, field_image_.cols / 2) - Vec2d(0.5, 0.5);  // 图像坐标

        if (pf[0] >= image_boarder_ && pf[0] < field_image_.cols - image_boarder_ && pf[1] >= image_boarder_ &&
            pf[1] < field_image_.rows - image_boarder_) {
            // 图像梯度
            float dx = 0.5 * (math::GetPixelValue<float>(field_image_, pf[0] + 1, pf[1]) -
                              math::GetPixelValue<float>(field_image_, pf[0] - 1, pf[1]));
            float dy = 0.5 * (math::GetPixelValue<float>(field_image_, pf[0], pf[1] + 1) -
                              math::GetPixelValue<float>(field_image_, pf[0], pf[1] - 1));

            _jacobianOplusXi << resolution_ * dx, resolution_ * dy,
                -resolution_ * dx * range_ * std::sin(angle_ + theta) +
                    resolution_ * dy * range_ * std::cos(angle_ + theta);
        } else {
            _jacobianOplusXi.setZero();
            setLevel(1);
        }
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
    const cv::Mat& field_image_;
    double range_ = 0;
    double angle_ = 0;
    float resolution_ = 10.0;
    inline static const int image_boarder_ = 10;
};

/**
 * TODO: 第五章习题1，P2P配准的Edge定义
 */
class EdgeSE2Point2Point : public g2o::BaseUnaryEdge<2, Vec2d, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Point2d = pcl::PointXY;
    using Cloud2d = pcl::PointCloud<Point2d>;

    EdgeSE2Point2Point(pcl::search::KdTree<Point2d>* kdtree, Cloud2d::ConstPtr target_cloud, double range, double angle)
        : kdtree_(kdtree), target_cloud_(target_cloud), range_(range), angle_(angle) {}

    void computeError() override {
        VertexSE2* v = (VertexSE2 *)_vertices[0];
        SE2 pose = v->estimate();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Point2d pt;
        pt.x = pw.x();
        pt.y = pw.y();

        nn_idx.clear();
        nn_dis.clear();
        kdtree_->nearestKSearch(pt, 1, nn_idx, nn_dis);

        if (nn_idx.size() > 0 && nn_dis[0] < max_dis2) {
            Point2d nn_pt = target_cloud_->points[nn_idx[0]];
            _error = Vec2d(pt.x - nn_pt.x, pt.y - nn_pt.y);
        } else {
            _error = Vec2d(0, 0);
            setLevel(1);
        }
    }

    void linearizeOplus() override {
        VertexSE2* v = (VertexSE2 *)_vertices[0];
        SE2 pose = v->estimate();
        float theta = pose.so2().log();

        if (nn_idx.size() > 0 && nn_dis[0] < max_dis2) {
            _jacobianOplusXi << 1, 0, -range_ * std::sin(angle_ + theta), 0, 1, range_ * std::cos(angle_ + theta);
        } else {
            _jacobianOplusXi.setZero();
            setLevel(1);
        }
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
    pcl::search::KdTree<Point2d>* kdtree_;
    std::vector<int> nn_idx;
    std::vector<float> nn_dis;
    Cloud2d::ConstPtr target_cloud_;
    double range_ = 0;
    double angle_ = 0;
    const float max_dis2 = 0.01;
};

/**
 * TODO: P2L配准的Edge定义
 */
class EdgeSE2Point2Line : public g2o::BaseUnaryEdge<1, double, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Point2d = pcl::PointXY;
    using Cloud2d = pcl::PointCloud<Point2d>;

    EdgeSE2Point2Line(pcl::search::KdTree<Point2d>* kdtree, Cloud2d::ConstPtr target_cloud, double range, double angle)
        : kdtree_(kdtree), target_cloud_(target_cloud), range_(range), angle_(angle) {}

    void computeError() override {
        VertexSE2* v = (VertexSE2 *)_vertices[0];
        SE2 pose = v->estimate();
        Vec2d pw = pose * Vec2d(range_ * std::cos(angle_), range_ * std::sin(angle_));
        Point2d pt;
        pt.x = pw.x();
        pt.y = pw.y();

        nn_idx.clear();
        nn_dis.clear();
        kdtree_->nearestKSearch(pt, 5, nn_idx, nn_dis);

        effective_pts.clear();
        for (int i = 0; i < nn_idx.size(); ++i) {
            if (nn_dis[i] < max_dis) {
                effective_pts.emplace_back(
                    Vec2d(target_cloud_->points[nn_idx[i]].x, target_cloud_->points[nn_idx[i]].y));
            }
        }

        if (effective_pts.size() < 3) {
            fit_success = false;
            _error[0] = 0;
            setLevel(1);
        } else {
            if (math::FitLine2D(effective_pts, line_coeffs)) {
                fit_success = true;
                _error[0] = line_coeffs[0] * pw.x() + line_coeffs[1] * pw.y() + line_coeffs[2];
            } else {
                fit_success = false;
                _error[0] = 0;
                setLevel(1);
            }
        }
    }

    void linearizeOplus() override {
        VertexSE2* v = (VertexSE2 *)_vertices[0];
        SE2 pose = v->estimate();
        float theta = pose.so2().log();

        if (!fit_success) {
            _jacobianOplusXi.setZero();
            setLevel(1);
        } else {
            _jacobianOplusXi << line_coeffs[0], line_coeffs[1],
                -line_coeffs[0] * range_ * std::sin(angle_ + theta) + line_coeffs[1] * range_ * std::cos(angle_ + theta);
        }
    }

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
    pcl::search::KdTree<Point2d>* kdtree_;
    std::vector<int> nn_idx;
    std::vector<float> nn_dis;
    Cloud2d::ConstPtr target_cloud_;
    std::vector<Vec2d> effective_pts;
    Vec3d line_coeffs;
    bool fit_success = false;
    double range_ = 0;
    double angle_ = 0;
    const float max_dis = 0.3;
};

/**
 * SE2 pose graph使用
 * error = v1.inv * v2 * meas.inv
 */
class EdgeSE2 : public g2o::BaseBinaryEdge<3, SE2, VertexSE2, VertexSE2> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2() {}

    void computeError() override {
        VertexSE2* v1 = (VertexSE2*)_vertices[0];
        VertexSE2* v2 = (VertexSE2*)_vertices[1];
        _error = (v1->estimate().inverse() * v2->estimate() * measurement().inverse()).log();
    }

    // TODO jacobian

    bool read(std::istream& is) override { return true; }
    bool write(std::ostream& os) const override { return true; }

   private:
};

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_G2O_TYPES_H
