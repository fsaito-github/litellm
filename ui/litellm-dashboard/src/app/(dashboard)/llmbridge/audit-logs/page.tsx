"use client";

import AuditLogViewer from "@/components/llmbridge/AuditLogViewer";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const AuditLogsPage = () => {
  const { accessToken } = useAuthorized();

  return <AuditLogViewer accessToken={accessToken} />;
};

export default AuditLogsPage;
