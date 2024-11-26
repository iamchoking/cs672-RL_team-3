//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"

size_t POGO_GC_PASSIVE_IDX = 6; // the gc index of the "passive" joint (needs pd gain set to spring const.)
size_t POGO_GC_PRISMATIC_IDX = 9; // the gc index of the "active" prismatic joint (needs different pd gain)


namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {

    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    pogo_ = world_->addArticulatedSystem(resourceDir_+"/pogo/urdf/pogo.urdf");
    pogo_->setName("pogo");
    pogo_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    auto* ground = world_->addGround();
    ground->setName("ground");

    /// get robot data
    gcDim_ = pogo_->getGeneralizedCoordinateDim();
    gvDim_ = pogo_->getDOF();
    nJoints_ = gvDim_ - 7; //idx 6 is the pogo stick passive joint

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget3_.setZero(nJoints_);

    /// this is nominal configuration of the pogo stick
    gc_init_ << 
      0, 0, 1,
      1.0, 0.0, 0.0, 0.0,
      0.0, 
      0.0, 0.0, 0.0;

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointDgain.setZero(); 

    double pGainRev, pGainPrsm;
    READ_YAML(double, pGainRev , cfg_["pogo_pgain_r_NM_rad"]); /// example of reading params from the config
    READ_YAML(double, pGainPrsm, cfg_["pogo_pgain_p_N_m"]);

    jointPgain.tail(nJoints_).setConstant(pGainRev);
    jointPgain[POGO_GC_PRISMATIC_IDX] = pGainPrsm;

    jointDgain.tail(nJoints_).setConstant(0.2);

    double pogoSpringConstant, pogoPreload;
    READ_YAML(double, pogoSpringConstant , cfg_["pogo_spring_const_N_M"]); /// example of reading params from the config
    READ_YAML(double, pogoPreload        , cfg_["pogo_preload_N"]);

    jointPgain[POGO_GC_PASSIVE_IDX] = pogoSpringConstant;
    pTarget_[POGO_GC_PASSIVE_IDX] = -(pogoPreload / pogoSpringConstant);

    // if(visualizable_){
    //   std::cout << "PD gain set as  : " << jointPgain.transpose() << std::endl;
    //   std::cout << "Initial Ptarget : " << pTarget_.transpose() << std::endl;
    // }

    pogo_->setPdGains(jointPgain, jointDgain);
    pogo_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 16; // search obDouble_
    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);

    /// visualize if it is the first environment
    if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(pogo_);
    }
  }

  void init() final { }

  void reset() final {
    pogo_->setState(gc_init_, gv_init_);
    updateObservation();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget3_ = action.cast<double>();
    pTarget3_ = pTarget3_.cwiseProduct(actionStd_);
    pTarget3_ += actionMean_;
    // if(visualizable_){std::cout << "recieved action: " << pTarget3_.transpose() << std::endl;}
    pTarget_.tail(nJoints_) = pTarget3_;

    pogo_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }
    pogo_->getState(gc_, gv_);

    // make sure recordReward comes right after updateObservation
    updateObservation();
    recordReward(&rewards_);

    return rewards_.sum();
  }

  inline void recordReward(Reward *rewards){
    // rewards_.record("torque", pogo_->getGeneralizedForce().squaredNorm());
    // rewards_.record("forwardVel", std::min(4.0, bodyLinearVel_[0]));

    rewards_.record("height", std::min(headPos_.e()[2],4.0)*0.25);
    rewards_.record("zvelup", std::max(0.0,std::min(gv_[2],10.0)*0.1));
  }

  void updateObservation() {
    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    pogo_->getFramePosition(pogo_->getFrameIdxByName("mass_head"),headPos_);
    // if(visualizable_){std::cout << "recorded head position: " << headPos_.e().transpose() << std::endl;}

    obDouble_ << gc_[2],                 /// body height      [0]
        rot.e().row(2).transpose(),      /// body orientation [1,2,3]
        gc_.tail(nJoints_),              /// joint angles     [4,5,6]
        bodyLinearVel_, bodyAngularVel_, /// body vel.        [7,8,9,10,11,12]
        gv_.tail(nJoints_);              /// joint velocity   [13,14,15]
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);



    //  if(visualizable_){
    //     for(auto n: pogo_->getBodyNames()){
    //     std::cout << n << ", ";
    //     }
    //     std::cout << std::endl;
    //  }
    for(auto& contact: pogo_->getContacts()){
      // terminal if "mass" and ground collide
      // if(visualizable_){
      //   std::cout << "analyzing contact " << contact.getIndexContactProblem() << std::endl;
      //   std::cout << "Pair Object: " << contact.getPairObjectIndex() << std::endl;
      //   std::cout << "Local Body : " << contact.getPairObjectIndex() << std::endl;

      //   std::cout << "Body Idx of [mass] : " << pogo_->getBodyIdx("mass") << std::endl;

      //   for(auto n: world_->getObjList()){
      //   std::cout << n->getName() << ", ";
      //   }
      //   std::cout << std::endl;

      //   std::cout << "Ground index in world: " << world_->getObject("ground")->getIndexInWorld() << std::endl;

      // }
      if(contact.getPairObjectIndex() == world_->getObject("ground")->getIndexInWorld() && contact.getlocalBodyIndex() == pogo_->getBodyIdx("mass")){
        // if(visualizable_){std::cout << "terminal (fall)" << std::endl;}
        return true;
      }
    }

    terminalReward = 0.f;
    return false;
  }

  void curriculumUpdate() { };

 private:
  size_t gcDim_, gvDim_, nJoints_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* pogo_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget3_, vTarget_;
  double terminalRewardCoeff_ = -10.;
  Eigen::VectorXd actionMean_, actionStd_, obDouble_;
  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;

  Vec<3> headPos_;

  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;
};
thread_local std::mt19937 raisim::ENVIRONMENT::gen_;

}


