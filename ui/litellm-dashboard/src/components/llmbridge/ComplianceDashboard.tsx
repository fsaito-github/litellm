"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Card,
  Typography,
  Select,
  Statistic,
  Row,
  Col,
  Badge,
  Spin,
  Tag,
  Descriptions,
  message,
} from "antd";
import {
  SafetyCertificateOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { BarChart } from "@tremor/react";
import {
  proxyBaseUrl,
} from "../networking";

const { Title } = Typography;

const globalLitellmHeaderName = "Authorization";

interface ComplianceDashboardProps {
  accessToken: string | null;
}

interface ComplianceStatus {
  enabled: boolean;
  pii_detection_enabled: boolean;
  data_residency_enabled: boolean;
  audit_logging_enabled: boolean;
  lgpd_mode: boolean;
  config_summary: Record<string, any>;
}

interface PIIDetection {
  type: string;
  count: number;
}

interface ComplianceReport {
  pii_detections: PIIDetection[];
  data_residency_violations: number;
  audit_coverage_percentage: number;
  total_requests_scanned: number;
  total_pii_detected: number;
  blocked_requests: number;
}

const ComplianceDashboard: React.FC<ComplianceDashboardProps> = ({
  accessToken,
}) => {
  const [period, setPeriod] = useState("30d");
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState<ComplianceStatus | null>(null);
  const [report, setReport] = useState<ComplianceReport | null>(null);

  const fetchData = useCallback(async () => {
    if (!accessToken) return;
    setLoading(true);

    const headers: Record<string, string> = {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    };
    const base = proxyBaseUrl ?? "";

    try {
      const [statusRes, reportRes] = await Promise.all([
        fetch(`${base}/compliance/latam/status`, {
          method: "GET",
          headers,
        }),
        fetch(`${base}/compliance/latam/report?period=${period}`, {
          method: "GET",
          headers,
        }),
      ]);

      if (!statusRes.ok || !reportRes.ok) {
        throw new Error("Failed to fetch compliance data");
      }

      const [statusData, reportData] = await Promise.all([
        statusRes.json(),
        reportRes.json(),
      ]);

      setStatus(statusData);
      setReport(reportData);
    } catch (err: any) {
      console.error("Compliance fetch error:", err);
      message.error(err.message ?? "Failed to load compliance data");
    } finally {
      setLoading(false);
    }
  }, [accessToken, period]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const featureTag = (enabled: boolean, label: string) => (
    <Tag
      icon={enabled ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
      color={enabled ? "success" : "default"}
    >
      {label}
    </Tag>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center" style={{ minHeight: 400 }}>
        <Spin size="large" tip="Loading compliance data..." />
      </div>
    );
  }

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <Title level={3} style={{ margin: 0 }}>
          <SafetyCertificateOutlined className="mr-2" />
          Compliance Dashboard
        </Title>
        <Select
          value={period}
          onChange={setPeriod}
          style={{ width: 140 }}
          options={[
            { label: "Last 7 days", value: "7d" },
            { label: "Last 30 days", value: "30d" },
            { label: "Last 90 days", value: "90d" },
          ]}
        />
      </div>

      {/* Compliance Status */}
      <Card className="mb-6" title="Compliance Status">
        <div className="flex items-center gap-4 mb-4">
          <Badge
            status={status?.enabled ? "success" : "error"}
            text={
              <span className="text-base font-medium">
                {status?.enabled
                  ? "Compliance Module Enabled"
                  : "Compliance Module Disabled"}
              </span>
            }
          />
        </div>
        <div className="flex flex-wrap gap-2 mb-4">
          {featureTag(status?.pii_detection_enabled ?? false, "PII Detection")}
          {featureTag(
            status?.data_residency_enabled ?? false,
            "Data Residency",
          )}
          {featureTag(
            status?.audit_logging_enabled ?? false,
            "Audit Logging",
          )}
          {featureTag(status?.lgpd_mode ?? false, "LGPD Mode")}
        </div>
        {status?.config_summary &&
          Object.keys(status.config_summary).length > 0 && (
            <Descriptions size="small" column={2} bordered>
              {Object.entries(status.config_summary).map(([key, value]) => (
                <Descriptions.Item key={key} label={key}>
                  {typeof value === "boolean" ? (
                    value ? (
                      <Tag color="green">Yes</Tag>
                    ) : (
                      <Tag>No</Tag>
                    )
                  ) : (
                    String(value)
                  )}
                </Descriptions.Item>
              ))}
            </Descriptions>
          )}
      </Card>

      {/* KPI Cards */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total PII Detected"
              value={report?.total_pii_detected ?? 0}
              valueStyle={{ color: "#fa8c16" }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Data Residency Violations"
              value={report?.data_residency_violations ?? 0}
              valueStyle={{
                color:
                  (report?.data_residency_violations ?? 0) > 0
                    ? "#cf1322"
                    : "#3f8600",
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Audit Coverage"
              value={report?.audit_coverage_percentage ?? 0}
              precision={1}
              suffix="%"
              valueStyle={{ color: "#1677ff" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Blocked Requests"
              value={report?.blocked_requests ?? 0}
              valueStyle={{ color: "#cf1322" }}
            />
          </Card>
        </Col>
      </Row>

      {/* PII Detections by Type */}
      <Card className="mb-6" title="PII Detections by Type">
        {report?.pii_detections && report.pii_detections.length > 0 ? (
          <BarChart
            data={report.pii_detections}
            index="type"
            categories={["count"]}
            colors={["amber"]}
            valueFormatter={(v) => String(v)}
            yAxisWidth={60}
            className="h-72"
            showAnimation
          />
        ) : (
          <div className="text-center text-gray-400 py-12">
            No PII detections in this period
          </div>
        )}
      </Card>

      {/* Scanned Requests */}
      <Card title="Scan Overview">
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12}>
            <Statistic
              title="Total Requests Scanned"
              value={report?.total_requests_scanned ?? 0}
            />
          </Col>
          <Col xs={24} sm={12}>
            <Statistic
              title="PII Detection Rate"
              value={
                report?.total_requests_scanned
                  ? (
                      ((report?.total_pii_detected ?? 0) /
                        report.total_requests_scanned) *
                      100
                    ).toFixed(2)
                  : 0
              }
              suffix="%"
            />
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default ComplianceDashboard;
