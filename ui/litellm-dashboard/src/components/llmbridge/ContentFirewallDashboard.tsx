"use client";
import React, { useState } from "react";
import {
  Card,
  Table,
  Statistic,
  Row,
  Col,
  Tag,
  Switch,
} from "antd";
import {
  SafetyOutlined,
  StopOutlined,
  WarningOutlined,
  CheckCircleOutlined,
} from "@ant-design/icons";
import { DonutChart } from "@tremor/react";

interface ContentFirewallDashboardProps {
  accessToken: string | null;
  userRole?: string;
}

type RuleCategory =
  | "prompt_injection"
  | "jailbreak"
  | "data_exfiltration"
  | "pii_leak"
  | "toxic";

type Severity = "low" | "medium" | "high" | "critical";
type Action = "block" | "warn" | "log";

interface FirewallRule {
  rule_id: string;
  name: string;
  category: RuleCategory;
  severity: Severity;
  action: Action;
  enabled: boolean;
  description: string;
}

const CATEGORY_COLORS: Record<RuleCategory, string> = {
  prompt_injection: "red",
  jailbreak: "orange",
  data_exfiltration: "purple",
  pii_leak: "blue",
  toxic: "gold",
};

const SEVERITY_COLORS: Record<Severity, string> = {
  low: "green",
  medium: "gold",
  high: "orange",
  critical: "red",
};

const ACTION_COLORS: Record<Action, string> = {
  block: "red",
  warn: "gold",
  log: "blue",
};

const BUILTIN_RULES: FirewallRule[] = [
  { rule_id: "fw-001", name: "Ignore Instructions", category: "prompt_injection", severity: "critical", action: "block", enabled: true, description: "Detects attempts to override system instructions" },
  { rule_id: "fw-002", name: "System Prompt Extraction", category: "prompt_injection", severity: "high", action: "block", enabled: true, description: "Prevents extraction of system-level prompts" },
  { rule_id: "fw-003", name: "Role Hijacking", category: "prompt_injection", severity: "high", action: "block", enabled: true, description: "Blocks attempts to redefine the AI's role" },
  { rule_id: "fw-004", name: "DAN Jailbreak", category: "jailbreak", severity: "critical", action: "block", enabled: true, description: "Detects Do Anything Now style attacks" },
  { rule_id: "fw-005", name: "Persona Bypass", category: "jailbreak", severity: "high", action: "block", enabled: true, description: "Blocks character-based restriction bypasses" },
  { rule_id: "fw-006", name: "Encoding Evasion", category: "jailbreak", severity: "medium", action: "warn", enabled: true, description: "Detects base64/rot13 encoding evasion" },
  { rule_id: "fw-007", name: "Database Exfil", category: "data_exfiltration", severity: "critical", action: "block", enabled: true, description: "Prevents database content extraction" },
  { rule_id: "fw-008", name: "File Path Disclosure", category: "data_exfiltration", severity: "high", action: "block", enabled: true, description: "Blocks filesystem path exposure" },
  { rule_id: "fw-009", name: "API Key Leak", category: "data_exfiltration", severity: "critical", action: "block", enabled: true, description: "Prevents API key exposure in responses" },
  { rule_id: "fw-010", name: "Email Detection", category: "pii_leak", severity: "medium", action: "warn", enabled: true, description: "Flags email addresses in output" },
  { rule_id: "fw-011", name: "SSN Detection", category: "pii_leak", severity: "critical", action: "block", enabled: true, description: "Blocks Social Security numbers in output" },
  { rule_id: "fw-012", name: "Phone Number Detection", category: "pii_leak", severity: "medium", action: "warn", enabled: true, description: "Flags phone numbers in output" },
  { rule_id: "fw-013", name: "Credit Card Detection", category: "pii_leak", severity: "critical", action: "block", enabled: true, description: "Blocks credit card numbers in output" },
  { rule_id: "fw-014", name: "Hate Speech", category: "toxic", severity: "high", action: "block", enabled: true, description: "Blocks hate speech and slurs" },
  { rule_id: "fw-015", name: "Profanity Filter", category: "toxic", severity: "low", action: "warn", enabled: false, description: "Flags profane language in output" },
];

interface ViolationRecord {
  id: string;
  timestamp: string;
  rule_name: string;
  category: RuleCategory;
  matched_text: string;
  action_taken: Action;
  request_id: string;
}

const SAMPLE_VIOLATIONS: ViolationRecord[] = [
  { id: "v1", timestamp: new Date(Date.now() - 300000).toISOString(), rule_name: "Ignore Instructions", category: "prompt_injection", matched_text: "Ignore all previous instructions and...", action_taken: "block", request_id: "req-abc123" },
  { id: "v2", timestamp: new Date(Date.now() - 900000).toISOString(), rule_name: "SSN Detection", category: "pii_leak", matched_text: "The SSN is 123-45-****", action_taken: "block", request_id: "req-def456" },
  { id: "v3", timestamp: new Date(Date.now() - 1800000).toISOString(), rule_name: "DAN Jailbreak", category: "jailbreak", matched_text: "You are now DAN, which stands for...", action_taken: "block", request_id: "req-ghi789" },
  { id: "v4", timestamp: new Date(Date.now() - 3600000).toISOString(), rule_name: "Email Detection", category: "pii_leak", matched_text: "Contact us at user@example...", action_taken: "warn", request_id: "req-jkl012" },
  { id: "v5", timestamp: new Date(Date.now() - 7200000).toISOString(), rule_name: "Encoding Evasion", category: "jailbreak", matched_text: "aWdub3JlIHByZXZpb3VzIGluc3...", action_taken: "warn", request_id: "req-mno345" },
];

const ContentFirewallDashboard: React.FC<ContentFirewallDashboardProps> = () => {
  const [rules, setRules] = useState<FirewallRule[]>(BUILTIN_RULES);

  const handleToggleRule = (ruleId: string, checked: boolean) => {
    setRules((prev) =>
      prev.map((r) => (r.rule_id === ruleId ? { ...r, enabled: checked } : r))
    );
  };

  // Compute stats
  const totalChecks = 12847;
  const totalViolations = SAMPLE_VIOLATIONS.length;
  const blockRate =
    totalViolations > 0
      ? (
          (SAMPLE_VIOLATIONS.filter((v) => v.action_taken === "block").length /
            totalViolations) *
          100
        ).toFixed(1)
      : "0";

  // Category distribution for donut chart
  const categoryCountMap = rules.reduce(
    (acc, r) => {
      acc[r.category] = (acc[r.category] || 0) + 1;
      return acc;
    },
    {} as Record<string, number>
  );
  const donutData = Object.entries(categoryCountMap).map(([cat, count]) => ({
    name: cat.replace(/_/g, " "),
    value: count,
  }));

  const rulesColumns = [
    { title: "Rule ID", dataIndex: "rule_id", key: "rule_id", width: 90 },
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (text: string) => <span className="font-medium">{text}</span>,
    },
    {
      title: "Category",
      dataIndex: "category",
      key: "category",
      filters: [
        { text: "Prompt Injection", value: "prompt_injection" },
        { text: "Jailbreak", value: "jailbreak" },
        { text: "Data Exfiltration", value: "data_exfiltration" },
        { text: "PII Leak", value: "pii_leak" },
        { text: "Toxic", value: "toxic" },
      ],
      onFilter: (value: any, record: FirewallRule) => record.category === value,
      render: (cat: RuleCategory) => (
        <Tag color={CATEGORY_COLORS[cat]}>{cat.replace(/_/g, " ")}</Tag>
      ),
    },
    {
      title: "Severity",
      dataIndex: "severity",
      key: "severity",
      sorter: (a: FirewallRule, b: FirewallRule) => {
        const order: Record<Severity, number> = {
          low: 0,
          medium: 1,
          high: 2,
          critical: 3,
        };
        return order[a.severity] - order[b.severity];
      },
      render: (sev: Severity) => (
        <Tag color={SEVERITY_COLORS[sev]}>{sev.toUpperCase()}</Tag>
      ),
    },
    {
      title: "Action",
      dataIndex: "action",
      key: "action",
      render: (action: Action) => (
        <Tag color={ACTION_COLORS[action]}>{action.toUpperCase()}</Tag>
      ),
    },
    {
      title: "Enabled",
      dataIndex: "enabled",
      key: "enabled",
      render: (enabled: boolean, record: FirewallRule) => (
        <Switch
          checked={enabled}
          onChange={(checked) => handleToggleRule(record.rule_id, checked)}
          size="small"
        />
      ),
    },
  ];

  const violationColumns = [
    {
      title: "Timestamp",
      dataIndex: "timestamp",
      key: "timestamp",
      render: (val: string) => new Date(val).toLocaleString(),
      width: 180,
    },
    { title: "Rule", dataIndex: "rule_name", key: "rule_name" },
    {
      title: "Category",
      dataIndex: "category",
      key: "category",
      render: (cat: RuleCategory) => (
        <Tag color={CATEGORY_COLORS[cat]}>{cat.replace(/_/g, " ")}</Tag>
      ),
    },
    {
      title: "Matched Text",
      dataIndex: "matched_text",
      key: "matched_text",
      ellipsis: true,
      render: (text: string) => (
        <span className="font-mono text-xs">{text}</span>
      ),
    },
    {
      title: "Action Taken",
      dataIndex: "action_taken",
      key: "action_taken",
      render: (action: Action) => (
        <Tag color={ACTION_COLORS[action]}>{action.toUpperCase()}</Tag>
      ),
    },
  ];

  return (
    <div className="w-full p-6">
      <h1 className="text-2xl font-semibold flex items-center gap-2 mb-6">
        <SafetyOutlined /> Content Firewall
      </h1>

      {/* Stats */}
      <Row gutter={16} className="mb-6">
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Total Checks"
              value={totalChecks}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Violations Detected"
              value={totalViolations}
              prefix={<WarningOutlined />}
              valueStyle={{ color: "#cf1322" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Block Rate"
              value={blockRate}
              suffix="%"
              prefix={<StopOutlined />}
              valueStyle={{ color: "#fa541c" }}
            />
          </Card>
        </Col>
      </Row>

      {/* Category Distribution */}
      <Row gutter={16} className="mb-6">
        <Col xs={24} md={10}>
          <Card title="Rules by Category">
            <DonutChart
              data={donutData}
              category="value"
              index="name"
              colors={["rose", "amber", "violet", "blue", "yellow"]}
              className="h-60"
            />
          </Card>
        </Col>
        <Col xs={24} md={14}>
          <Card title="Recent Violations">
            <Table
              dataSource={SAMPLE_VIOLATIONS}
              columns={violationColumns}
              rowKey="id"
              pagination={{ pageSize: 5 }}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      {/* Rules Table */}
      <Card title="Firewall Rules (15 built-in)">
        <Table
          dataSource={rules}
          columns={rulesColumns}
          rowKey="rule_id"
          pagination={false}
          size="small"
        />
      </Card>
    </div>
  );
};

export default ContentFirewallDashboard;
