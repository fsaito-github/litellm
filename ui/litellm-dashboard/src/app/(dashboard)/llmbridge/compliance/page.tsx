"use client";

import ComplianceDashboard from "@/components/llmbridge/ComplianceDashboard";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const CompliancePage = () => {
  const { accessToken } = useAuthorized();

  return <ComplianceDashboard accessToken={accessToken} />;
};

export default CompliancePage;
