"use client";

import FinOpsDashboard from "@/components/llmbridge/FinOpsDashboard";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const FinOpsPage = () => {
  const { accessToken } = useAuthorized();

  return <FinOpsDashboard accessToken={accessToken} />;
};

export default FinOpsPage;
