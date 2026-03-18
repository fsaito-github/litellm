"use client";

import React from "react";
import { Select } from "antd";
import { CloudServerOutlined } from "@ant-design/icons";
import { useWorker } from "@/hooks/useWorker";

interface WorkerDropdownProps {
  onLogout: () => void;
}

const WorkerDropdown: React.FC<WorkerDropdownProps> = ({ onLogout }) => {
  const { isControlPlane, selectedWorker, workers } = useWorker();

  if (!isControlPlane || !selectedWorker) return null;

  return (
    <Select
      value={selectedWorker.worker_id}
      style={{ minWidth: 180 }}
      suffixIcon={<CloudServerOutlined />}
      options={workers.map((w) => ({
        label: w.name,
        value: w.worker_id,
        disabled: w.worker_id === selectedWorker.worker_id,
      }))}
      onChange={() => {
        // Behaves exactly like logout — clears session, SSO logout, etc.
        // User re-authenticates on the login page and picks the new worker there.
        onLogout();
      }}
    />
  );
};

export default WorkerDropdown;
