//
// Created by xiang on 23-1-19.
//

#include "ch4/g2o_types.h"
#include "common/g2o_types.h"

#include <glog/logging.h>

namespace sad {

EdgeInertial::EdgeInertial(std::shared_ptr<IMUPreintegration> preinteg, const Vec3d& gravity, double weight)
    : preint_(preinteg), dt_(preinteg->dt_) {
    resize(6);  // 6个关联顶点
    grav_ = gravity;
    setInformation(preinteg->cov_.inverse() * weight);
}

void EdgeInertial::computeError() {
    auto* p1 = dynamic_cast<const VertexPose*>(_vertices[0]);
    auto* v1 = dynamic_cast<const VertexVelocity*>(_vertices[1]);
    auto* bg1 = dynamic_cast<const VertexGyroBias*>(_vertices[2]);
    auto* ba1 = dynamic_cast<const VertexAccBias*>(_vertices[3]);
    auto* p2 = dynamic_cast<const VertexPose*>(_vertices[4]);
    auto* v2 = dynamic_cast<const VertexVelocity*>(_vertices[5]);

    Vec3d bg = bg1->estimate();
    Vec3d ba = ba1->estimate();

    const SO3 dR = preint_->GetDeltaRotation(bg);
    const Vec3d dv = preint_->GetDeltaVelocity(bg, ba);
    const Vec3d dp = preint_->GetDeltaPosition(bg, ba);

    /// 预积分误差项（4.41）
    const Vec3d er = (dR.inverse() * p1->estimate().so3().inverse() * p2->estimate().so3()).log();
    Mat3d RiT = p1->estimate().so3().inverse().matrix();
    const Vec3d ev = RiT * (v2->estimate() - v1->estimate() - grav_ * dt_) - dv;
    const Vec3d ep = RiT * (p2->estimate().translation() - p1->estimate().translation() - v1->estimate() * dt_ -
                            grav_ * dt_ * dt_ / 2) -
                     dp;
    _error << er, ev, ep;
}

void EdgeInertial::linearizeOplus() {
    auto* p1 = dynamic_cast<const VertexPose*>(_vertices[0]);
    auto* v1 = dynamic_cast<const VertexVelocity*>(_vertices[1]);
    auto* bg1 = dynamic_cast<const VertexGyroBias*>(_vertices[2]);
    auto* ba1 = dynamic_cast<const VertexAccBias*>(_vertices[3]);
    auto* p2 = dynamic_cast<const VertexPose*>(_vertices[4]);
    auto* v2 = dynamic_cast<const VertexVelocity*>(_vertices[5]);

    Vec3d bg = bg1->estimate();
    Vec3d ba = ba1->estimate();
    Vec3d dbg = bg - preint_->bg_;

    // 一些中间符号
    const SO3 R1 = p1->estimate().so3();
    const SO3 R1T = R1.inverse();
    const SO3 R2 = p2->estimate().so3();

    auto dR_dbg = preint_->dR_dbg_;
    auto dv_dbg = preint_->dV_dbg_;
    auto dp_dbg = preint_->dP_dbg_;
    auto dv_dba = preint_->dV_dba_;
    auto dp_dba = preint_->dP_dba_;

    // 估计值
    Vec3d vi = v1->estimate();
    Vec3d vj = v2->estimate();
    Vec3d pi = p1->estimate().translation();
    Vec3d pj = p2->estimate().translation();

    const SO3 dR = preint_->GetDeltaRotation(bg);
    const SO3 eR = SO3(dR).inverse() * R1T * R2;
    const Vec3d er = eR.log();
    const Mat3d invJr = SO3::jr_inv(eR);

    /// 雅可比矩阵
    /// 注意有3个index, 顶点的，自己误差的，顶点内部变量的
    /// 变量顺序：pose1(R1,p1), v1, bg1, ba1, pose2(R2,p2), v2
    /// 残差顺序：eR, ev, ep，残差顺序为行，变量顺序为列

    //       | R1 | p1 | v1 | bg1 | ba1 | R2 | p2 | v2 |
    //  vert | 0       | 1  | 2   | 3   | 4       | 5  |
    //  col  | 0    3  | 0  | 0   | 0   | 0    3  | 0  |
    //    row
    //  eR 0 |
    //  ev 3 |
    //  ep 6 |

    /// 残差对R1, 9x3
    _jacobianOplus[0].setZero();
    // dR/dR1, 4.42
    _jacobianOplus[0].block<3, 3>(0, 0) = -invJr * (R2.inverse() * R1).matrix();
    // dv/dR1, 4.47
    _jacobianOplus[0].block<3, 3>(3, 0) = SO3::hat(R1T * (vj - vi - grav_ * dt_));
    // dp/dR1, 4.48d
    _jacobianOplus[0].block<3, 3>(6, 0) = SO3::hat(R1T * (pj - pi - v1->estimate() * dt_ - 0.5 * grav_ * dt_ * dt_));

    /// 残差对p1, 9x3
    // dp/dp1, 4.48a
    _jacobianOplus[0].block<3, 3>(6, 3) = -R1T.matrix();

    /// 残差对v1, 9x3
    _jacobianOplus[1].setZero();
    // dv/dv1, 4.46a
    _jacobianOplus[1].block<3, 3>(3, 0) = -R1T.matrix();
    // dp/dv1, 4.48c
    _jacobianOplus[1].block<3, 3>(6, 0) = -R1T.matrix() * dt_;

    /// 残差对bg1
    _jacobianOplus[2].setZero();
    // dR/dbg1, 4.45
    _jacobianOplus[2].block<3, 3>(0, 0) = -invJr * eR.inverse().matrix() * SO3::jr((dR_dbg * dbg).eval()) * dR_dbg;
    // dv/dbg1
    _jacobianOplus[2].block<3, 3>(3, 0) = -dv_dbg;
    // dp/dbg1
    _jacobianOplus[2].block<3, 3>(6, 0) = -dp_dbg;

    /// 残差对ba1
    _jacobianOplus[3].setZero();
    // dv/dba1
    _jacobianOplus[3].block<3, 3>(3, 0) = -dv_dba;
    // dp/dba1
    _jacobianOplus[3].block<3, 3>(6, 0) = -dp_dba;

    /// 残差对pose2
    _jacobianOplus[4].setZero();
    // dr/dr2, 4.43
    _jacobianOplus[4].block<3, 3>(0, 0) = invJr;
    // dp/dp2, 4.48b
    _jacobianOplus[4].block<3, 3>(6, 3) = R1T.matrix();

    /// 残差对v2
    _jacobianOplus[5].setZero();
    // dv/dv2, 4,46b
    _jacobianOplus[5].block<3, 3>(3, 0) = R1T.matrix();  // OK

    // TODO: 深蓝学院第三章作业第三题 - 验证数值导和解析导的一致性
    // 验证了雅克比中较为复杂的残差对姿态的雅克比这部分，其余部分求数值导不需要使用广义加法增加扰动，较为容易
    {
        // 打印_jacobianOplus[0]的解析解雅克比
        LOG(INFO) << "[解析导] 残差对Posei(R_i p_i)的Jacobian: \n" << _jacobianOplus[0];

        // 定义一个小量
        const double eps = 1e-9;
        // 定义扰动
        Vec3d delta_x(eps, 0, 0);
        Vec3d delta_y(0, eps, 0);
        Vec3d delta_z(0, 0, eps);

        // 写出带有此扰动的状态量Ri和pi
        const SO3 Ri_updateX = R1 * SO3::exp(delta_x);
        const SO3 Ri_updateY = R1 * SO3::exp(delta_y);
        const SO3 Ri_updateZ = R1 * SO3::exp(delta_z);
        const Vec3d pi_updateX = pi + delta_x;
        const Vec3d pi_updateY = pi + delta_y;
        const Vec3d pi_updateZ = pi + delta_z;

        // 定义数值解雅克比
        Mat96d numerical_jacobian = Mat96d::Zero();

        // 更新前的rv
        const Vec3d ev = R1T.matrix() * (vj - vi - grav_ * dt_) - preint_->GetDeltaVelocity(bg, ba);
        // 计算出更新后的残差rR rv rp
        const Vec3d er_updateX = SO3(SO3(dR).inverse() * Ri_updateX.inverse() * R2).log(); // rR: 仅对Ri有雅克比，这里update也是指的Ri。对pi的数值雅克比残差项不会发生改变，所以雅克比是0
        const Vec3d er_updateY = SO3(SO3(dR).inverse() * Ri_updateY.inverse() * R2).log();
        const Vec3d er_updateZ = SO3(SO3(dR).inverse() * Ri_updateZ.inverse() * R2).log();
        const Vec3d ev_updateX = Ri_updateX.inverse().matrix() * (vj - vi - grav_ * dt_) - preint_->GetDeltaVelocity(bg, ba); // rv
        const Vec3d ev_updateY = Ri_updateY.inverse().matrix() * (vj - vi - grav_ * dt_) - preint_->GetDeltaVelocity(bg, ba);
        const Vec3d ev_updateZ = Ri_updateZ.inverse().matrix() * (vj - vi - grav_ * dt_) - preint_->GetDeltaVelocity(bg, ba);
        const Vec3d ep_updateRx = Ri_updateX.inverse().matrix() * (pj - pi - vi * dt_ - 0.5 * grav_ * dt_ * dt_) - preint_->GetDeltaPosition(bg, ba); // rp: update Ri
        const Vec3d ep_updateRy = Ri_updateY.inverse().matrix() * (pj - pi - vi * dt_ - 0.5 * grav_ * dt_ * dt_) - preint_->GetDeltaPosition(bg, ba);
        const Vec3d ep_updateRz = Ri_updateZ.inverse().matrix() * (pj - pi - vi * dt_ - 0.5 * grav_ * dt_ * dt_) - preint_->GetDeltaPosition(bg, ba);
        const Vec3d ep_updatepx = R1T * (pj - pi_updateX - vi * dt_ - 0.5 * grav_ * dt_ * dt_) - preint_->GetDeltaPosition(bg, ba); // rp: update pi
        const Vec3d ep_updatepy = R1T * (pj - pi_updateY - vi * dt_ - 0.5 * grav_ * dt_ * dt_) - preint_->GetDeltaPosition(bg, ba);
        const Vec3d ep_updatepz = R1T * (pj - pi_updateZ - vi * dt_ - 0.5 * grav_ * dt_ * dt_) - preint_->GetDeltaPosition(bg, ba);

        // 整理
        Vec9d e_updateRx, e_updateRy, e_updateRz, e_updatepx, e_updatepy, e_updatepz;
        e_updateRx << er_updateX, ev_updateX, ep_updateRx;
        e_updateRy << er_updateY, ev_updateY, ep_updateRy;
        e_updateRz << er_updateZ, ev_updateZ, ep_updateRz;
        e_updatepx << er, ev, ep_updatepx; // er ev定义式中不包含pi，无需更新
        e_updatepy << er, ev, ep_updatepy;
        e_updatepz << er, ev, ep_updatepz;

        // 填入数值解
        numerical_jacobian.col(0) = (e_updateRx - _error) / eps;
        numerical_jacobian.col(1) = (e_updateRy - _error) / eps;
        numerical_jacobian.col(2) = (e_updateRz - _error) / eps;
        numerical_jacobian.col(3) = (e_updatepx - _error) / eps;
        numerical_jacobian.col(4) = (e_updatepy - _error) / eps;
        numerical_jacobian.col(5) = (e_updatepz - _error) / eps;

        // 打印数值解雅克比
        LOG(INFO) << "[数值导] 残差对Posei(R_i p_i)的Jacobian: \n" << numerical_jacobian;
    }
}

}  // namespace sad