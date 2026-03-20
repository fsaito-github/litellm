"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import {
  Card,
  Typography,
  Table,
  Select,
  Input,
  DatePicker,
  Button,
  Tag,
  Space,
  Spin,
  Switch,
  Tooltip,
  message,
} from "antd";
import {
  FileSearchOutlined,
  DownloadOutlined,
  ReloadOutlined,
  SyncOutlined,
} from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import {
  proxyBaseUrl,
} from "../networking";

const { Title } = Typography;
const { RangePicker } = DatePicker;

const globalLitellmHeaderName = "Authorization";

interface AuditLogViewerProps {
  accessToken: string | null;
}

interface AuditEntry {
  id: string;
  timestamp: string;
  action: string;
  actor_id: string;
  actor_type: string;
  resource_type: string;
  resource_id: string;
  status: string;
  details: Record<string, any> | null;
}

interface AuditResponse {
  logs: AuditEntry[];
  total: number;
  page: number;
  page_size: number;
}

const STATUS_COLOR: Record<string, string> = {
  success: "green",
  failure: "red",
  blocked: "orange",
  error: "red",
  denied: "volcano",
};

const ACTION_OPTIONS = [
  { label: "All Actions", value: "" },
  { label: "Create", value: "create" },
  { label: "Update", value: "update" },
  { label: "Delete", value: "delete" },
  { label: "Read", value: "read" },
  { label: "Login", value: "login" },
  { label: "Logout", value: "logout" },
  { label: "API Call", value: "api_call" },
];

const RESOURCE_OPTIONS = [
  { label: "All Resources", value: "" },
  { label: "Key", value: "key" },
  { label: "Team", value: "team" },
  { label: "User", value: "user" },
  { label: "Model", value: "model" },
  { label: "Organization", value: "organization" },
  { label: "Budget", value: "budget" },
  { label: "Config", value: "config" },
];

const AuditLogViewer: React.FC<AuditLogViewerProps> = ({ accessToken }) => {
  const [loading, setLoading] = useState(true);
  const [logs, setLogs] = useState<AuditEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // Filters
  const [actorId, setActorId] = useState("");
  const [action, setAction] = useState("");
  const [resourceType, setResourceType] = useState("");
  const [status, setStatus] = useState("");
  const [dateRange, setDateRange] = useState<[string, string] | null>(null);

  // Auto-refresh
  const [autoRefresh, setAutoRefresh] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const buildQueryParams = useCallback(() => {
    const params = new URLSearchParams();
    params.set("page", String(page));
    params.set("page_size", String(pageSize));
    if (actorId) params.set("actor_id", actorId);
    if (action) params.set("action", action);
    if (resourceType) params.set("resource_type", resourceType);
    if (status) params.set("status", status);
    if (dateRange) {
      params.set("start_date", dateRange[0]);
      params.set("end_date", dateRange[1]);
    }
    return params.toString();
  }, [page, pageSize, actorId, action, resourceType, status, dateRange]);

  const fetchLogs = useCallback(async () => {
    if (!accessToken) return;
    setLoading(true);

    const headers: Record<string, string> = {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    };
    const base = proxyBaseUrl ?? "";
    const qs = buildQueryParams();

    try {
      const res = await fetch(`${base}/audit/logs?${qs}`, {
        method: "GET",
        headers,
      });

      if (!res.ok) {
        throw new Error("Failed to fetch audit logs");
      }

      const data: AuditResponse = await res.json();
      setLogs(data.logs ?? []);
      setTotal(data.total ?? 0);
    } catch (err: any) {
      console.error("Audit log fetch error:", err);
      message.error(err.message ?? "Failed to load audit logs");
    } finally {
      setLoading(false);
    }
  }, [accessToken, buildQueryParams]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh) {
      intervalRef.current = setInterval(fetchLogs, 30000);
    } else if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [autoRefresh, fetchLogs]);

  const handleExport = () => {
    if (!accessToken) return;
    const base = proxyBaseUrl ?? "";
    const qs = buildQueryParams();
    window.open(
      `${base}/audit/logs/export?${qs}&token=${encodeURIComponent(accessToken)}`,
      "_blank",
    );
  };

  const handleDateChange = (_: any, dateStrings: [string, string]) => {
    if (dateStrings[0] && dateStrings[1]) {
      setDateRange(dateStrings);
    } else {
      setDateRange(null);
    }
    setPage(1);
  };

  const columns: ColumnsType<AuditEntry> = [
    {
      title: "Timestamp",
      dataIndex: "timestamp",
      key: "timestamp",
      width: 180,
      render: (val: string) => {
        try {
          return new Date(val).toLocaleString();
        } catch {
          return val;
        }
      },
      sorter: (a, b) =>
        new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime(),
      defaultSortOrder: "descend",
    },
    {
      title: "Action",
      dataIndex: "action",
      key: "action",
      width: 120,
      render: (val: string) => (
        <Tag color="blue">{val}</Tag>
      ),
    },
    {
      title: "Actor ID",
      dataIndex: "actor_id",
      key: "actor_id",
      width: 160,
      ellipsis: true,
    },
    {
      title: "Actor Type",
      dataIndex: "actor_type",
      key: "actor_type",
      width: 110,
    },
    {
      title: "Resource Type",
      dataIndex: "resource_type",
      key: "resource_type",
      width: 130,
    },
    {
      title: "Resource ID",
      dataIndex: "resource_id",
      key: "resource_id",
      width: 160,
      ellipsis: true,
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      width: 100,
      render: (val: string) => (
        <Tag color={STATUS_COLOR[val?.toLowerCase()] ?? "default"}>
          {val}
        </Tag>
      ),
    },
    {
      title: "Details",
      dataIndex: "details",
      key: "details",
      ellipsis: true,
      render: (val: Record<string, any> | null) =>
        val ? JSON.stringify(val) : "-",
    },
  ];

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <Title level={3} style={{ margin: 0 }}>
          <FileSearchOutlined className="mr-2" />
          Audit Log Viewer
        </Title>
        <Space>
          <Tooltip title="Auto-refresh every 30s">
            <Switch
              checkedChildren={<SyncOutlined spin />}
              unCheckedChildren="Auto"
              checked={autoRefresh}
              onChange={setAutoRefresh}
            />
          </Tooltip>
          <Button icon={<ReloadOutlined />} onClick={fetchLogs}>
            Refresh
          </Button>
          <Button
            icon={<DownloadOutlined />}
            type="primary"
            onClick={handleExport}
          >
            Export CSV
          </Button>
        </Space>
      </div>

      {/* Filter Bar */}
      <Card className="mb-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          <Input
            placeholder="Actor ID"
            value={actorId}
            onChange={(e) => {
              setActorId(e.target.value);
              setPage(1);
            }}
            allowClear
          />
          <Select
            value={action}
            onChange={(val) => {
              setAction(val);
              setPage(1);
            }}
            options={ACTION_OPTIONS}
            style={{ width: "100%" }}
            placeholder="Action"
          />
          <Select
            value={resourceType}
            onChange={(val) => {
              setResourceType(val);
              setPage(1);
            }}
            options={RESOURCE_OPTIONS}
            style={{ width: "100%" }}
            placeholder="Resource Type"
          />
          <Select
            value={status}
            onChange={(val) => {
              setStatus(val);
              setPage(1);
            }}
            options={[
              { label: "All Statuses", value: "" },
              { label: "Success", value: "success" },
              { label: "Failure", value: "failure" },
              { label: "Blocked", value: "blocked" },
            ]}
            style={{ width: "100%" }}
            placeholder="Status"
          />
          <RangePicker
            onChange={handleDateChange}
            style={{ width: "100%" }}
          />
        </div>
      </Card>

      {/* Logs Table */}
      <Card>
        <Table
          columns={columns}
          dataSource={logs}
          rowKey="id"
          loading={loading}
          pagination={{
            current: page,
            pageSize,
            total,
            showSizeChanger: true,
            pageSizeOptions: ["10", "25", "50", "100"],
            showTotal: (t) => `Total ${t} entries`,
            onChange: (p, ps) => {
              setPage(p);
              setPageSize(ps);
            },
          }}
          scroll={{ x: 1100 }}
          size="small"
        />
      </Card>
    </div>
  );
};

export default AuditLogViewer;
