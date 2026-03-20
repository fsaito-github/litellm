"use client";
import React, { useState, useEffect, useCallback } from "react";
import {
  Card,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Statistic,
  Row,
  Col,
  Tag,
  message,
  Spin,
} from "antd";
import {
  RobotOutlined,
  PlusOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  DollarOutlined,
} from "@ant-design/icons";
import { BarChart } from "@tremor/react";
import { proxyBaseUrl } from "../networking";

interface AgentGatewayDashboardProps {
  accessToken: string | null;
  userRole?: string;
}

interface AgentRecord {
  agent_id: string;
  name: string;
  model: string;
  description: string;
  system_prompt?: string;
  invocations: number;
  avg_latency_ms: number;
  total_cost: number;
  status: string;
  created_at: string;
}

interface AgentStats {
  total_agents: number;
  active_sessions: number;
  total_invocations: number;
}

interface SessionRecord {
  session_id: string;
  agent_name: string;
  user_id: string;
  started_at: string;
  duration_ms: number;
  messages: number;
  status: string;
}

const AgentGatewayDashboard: React.FC<AgentGatewayDashboardProps> = ({
  accessToken,
}) => {
  const [agents, setAgents] = useState<AgentRecord[]>([]);
  const [stats, setStats] = useState<AgentStats>({
    total_agents: 0,
    active_sessions: 0,
    total_invocations: 0,
  });
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [registerModalOpen, setRegisterModalOpen] = useState(false);
  const [registerLoading, setRegisterLoading] = useState(false);
  const [form] = Form.useForm();

  const fetchAgents = useCallback(async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const url = proxyBaseUrl ? `${proxyBaseUrl}/agents` : `/agents`;
      const res = await fetch(url, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });
      if (res.ok) {
        const data = await res.json();
        const agentList: AgentRecord[] = (data.agents || data || []).map(
          (a: any, idx: number) => ({
            agent_id: a.agent_id || a.id || `agent-${idx}`,
            name: a.agent_name || a.name || "Unknown",
            model: a.litellm_model_id || a.model || "—",
            description: a.description || "",
            system_prompt: a.system_prompt || "",
            invocations: a.invocations ?? 0,
            avg_latency_ms: a.avg_latency_ms ?? 0,
            total_cost: a.total_cost ?? 0,
            status: a.status || "active",
            created_at: a.created_at || "",
          })
        );
        setAgents(agentList);
      }
    } catch (err) {
      console.error("Error fetching agents:", err);
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  const fetchStats = useCallback(async () => {
    if (!accessToken) return;
    try {
      const url = proxyBaseUrl
        ? `${proxyBaseUrl}/agents/stats`
        : `/agents/stats`;
      const res = await fetch(url, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });
      if (res.ok) {
        const data = await res.json();
        setStats({
          total_agents: data.total_agents ?? agents.length,
          active_sessions: data.active_sessions ?? 0,
          total_invocations: data.total_invocations ?? 0,
        });
        if (data.recent_sessions) {
          setSessions(data.recent_sessions);
        }
      }
    } catch {
      // Stats endpoint may not exist yet — use agent list count
      setStats((prev) => ({ ...prev, total_agents: agents.length }));
    }
  }, [accessToken, agents.length]);

  useEffect(() => {
    fetchAgents();
  }, [fetchAgents]);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  const handleRegister = async (values: any) => {
    if (!accessToken) return;
    setRegisterLoading(true);
    try {
      const url = proxyBaseUrl
        ? `${proxyBaseUrl}/agents/register`
        : `/agents/register`;
      const res = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          agent_name: values.name,
          description: values.description,
          model: values.model,
          system_prompt: values.system_prompt,
        }),
      });
      if (res.ok) {
        message.success("Agent registered successfully");
        setRegisterModalOpen(false);
        form.resetFields();
        fetchAgents();
      } else {
        const err = await res.json().catch(() => ({}));
        message.error(err.detail || "Failed to register agent");
      }
    } catch (err) {
      message.error("Network error registering agent");
    } finally {
      setRegisterLoading(false);
    }
  };

  // Chart data
  const chartData = agents
    .filter((a) => a.invocations > 0)
    .sort((a, b) => b.invocations - a.invocations)
    .slice(0, 10)
    .map((a) => ({
      agent: a.name,
      Invocations: a.invocations,
    }));

  const agentColumns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (text: string) => <span className="font-medium">{text}</span>,
    },
    {
      title: "Model",
      dataIndex: "model",
      key: "model",
      render: (model: string) => <Tag color="blue">{model}</Tag>,
    },
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
      ellipsis: true,
    },
    {
      title: "Invocations",
      dataIndex: "invocations",
      key: "invocations",
      sorter: (a: AgentRecord, b: AgentRecord) =>
        a.invocations - b.invocations,
      render: (val: number) => val.toLocaleString(),
    },
    {
      title: "Avg Latency",
      dataIndex: "avg_latency_ms",
      key: "avg_latency_ms",
      render: (val: number) => (val > 0 ? `${val.toFixed(0)} ms` : "—"),
    },
    {
      title: "Cost",
      dataIndex: "total_cost",
      key: "total_cost",
      render: (val: number) => (val > 0 ? `$${val.toFixed(4)}` : "—"),
    },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status: string) => (
        <Tag color={status === "active" ? "green" : "default"}>{status}</Tag>
      ),
    },
  ];

  const sessionColumns = [
    { title: "Session ID", dataIndex: "session_id", key: "session_id", ellipsis: true },
    { title: "Agent", dataIndex: "agent_name", key: "agent_name" },
    { title: "User", dataIndex: "user_id", key: "user_id", ellipsis: true },
    {
      title: "Started",
      dataIndex: "started_at",
      key: "started_at",
      render: (val: string) =>
        val ? new Date(val).toLocaleString() : "—",
    },
    {
      title: "Duration",
      dataIndex: "duration_ms",
      key: "duration_ms",
      render: (val: number) =>
        val > 0 ? `${(val / 1000).toFixed(1)}s` : "—",
    },
    { title: "Messages", dataIndex: "messages", key: "messages" },
    {
      title: "Status",
      dataIndex: "status",
      key: "status",
      render: (status: string) => {
        const color =
          status === "active"
            ? "green"
            : status === "completed"
              ? "blue"
              : "default";
        return <Tag color={color}>{status}</Tag>;
      },
    },
  ];

  return (
    <div className="w-full p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-semibold flex items-center gap-2">
          <RobotOutlined /> Agent Gateway
        </h1>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setRegisterModalOpen(true)}
        >
          Register Agent
        </Button>
      </div>

      {/* Stats */}
      <Row gutter={16} className="mb-6">
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Total Agents"
              value={stats.total_agents}
              prefix={<RobotOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Active Sessions"
              value={stats.active_sessions}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: "#3f8600" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="Total Invocations"
              value={stats.total_invocations}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* Agents Table */}
      <Card title="Registered Agents" className="mb-6">
        <Table
          dataSource={agents}
          columns={agentColumns}
          rowKey="agent_id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>

      {/* Usage Chart */}
      {chartData.length > 0 && (
        <Card title="Invocations by Agent" className="mb-6">
          <BarChart
            data={chartData}
            index="agent"
            categories={["Invocations"]}
            colors={["blue"]}
            yAxisWidth={60}
            className="h-72"
          />
        </Card>
      )}

      {/* Recent Sessions */}
      <Card title="Recent Sessions">
        <Table
          dataSource={sessions}
          columns={sessionColumns}
          rowKey="session_id"
          pagination={{ pageSize: 5 }}
          size="small"
          locale={{ emptyText: "No recent sessions" }}
        />
      </Card>

      {/* Register Agent Modal */}
      <Modal
        title="Register New Agent"
        open={registerModalOpen}
        onCancel={() => {
          setRegisterModalOpen(false);
          form.resetFields();
        }}
        footer={null}
        destroyOnClose
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleRegister}
        >
          <Form.Item
            name="name"
            label="Agent Name"
            rules={[{ required: true, message: "Please enter agent name" }]}
          >
            <Input placeholder="my-agent" />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input placeholder="What does this agent do?" />
          </Form.Item>
          <Form.Item
            name="model"
            label="Model"
            rules={[{ required: true, message: "Please select a model" }]}
          >
            <Select placeholder="Select model">
              <Select.Option value="gpt-4">gpt-4</Select.Option>
              <Select.Option value="gpt-4o">gpt-4o</Select.Option>
              <Select.Option value="gpt-3.5-turbo">gpt-3.5-turbo</Select.Option>
              <Select.Option value="claude-3-opus">claude-3-opus</Select.Option>
              <Select.Option value="claude-3-sonnet">claude-3-sonnet</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="system_prompt" label="System Prompt">
            <Input.TextArea
              rows={4}
              placeholder="You are a helpful assistant..."
            />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={registerLoading}
              block
            >
              Register Agent
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default AgentGatewayDashboard;
