"use client";
import React, { useState, useEffect, useCallback } from "react";
import {
  Card,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Tag,
  message,
  Spin,
  Descriptions,
  Empty,
  Popconfirm,
} from "antd";
import {
  FileTextOutlined,
  PlusOutlined,
  HistoryOutlined,
  PlayCircleOutlined,
  RollbackOutlined,
} from "@ant-design/icons";
import { proxyBaseUrl } from "../networking";

interface PromptManagementDashboardProps {
  accessToken: string | null;
  userRole?: string;
}

interface PromptRecord {
  prompt_id: string;
  name: string;
  description: string;
  template: string;
  version: number;
  variables: string[];
  created_by: string;
  created_at: string;
  updated_at: string;
}

interface VersionRecord {
  version: number;
  template: string;
  created_at: string;
  created_by: string;
}

function extractVariables(template: string): string[] {
  const matches = template.match(/\{\{(\w+)\}\}/g);
  if (!matches) return [];
  return [...new Set(matches.map((m) => m.replace(/\{\{|\}\}/g, "")))];
}

const PromptManagementDashboard: React.FC<PromptManagementDashboardProps> = ({
  accessToken,
}) => {
  const [prompts, setPrompts] = useState<PromptRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const [form] = Form.useForm();

  // Version history state
  const [versions, setVersions] = useState<Record<string, VersionRecord[]>>({});
  const [versionsLoading, setVersionsLoading] = useState<Record<string, boolean>>({});

  // Render test state
  const [renderModalOpen, setRenderModalOpen] = useState(false);
  const [renderPrompt, setRenderPrompt] = useState<PromptRecord | null>(null);
  const [renderVariables, setRenderVariables] = useState<Record<string, string>>({});
  const [renderOutput, setRenderOutput] = useState<string>("");
  const [renderLoading, setRenderLoading] = useState(false);

  const fetchPrompts = useCallback(async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const url = proxyBaseUrl ? `${proxyBaseUrl}/prompts` : `/prompts`;
      const res = await fetch(url, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });
      if (res.ok) {
        const data = await res.json();
        const promptList: PromptRecord[] = (
          data.prompts || data || []
        ).map((p: any) => ({
          prompt_id: p.prompt_id || p.id || p.name,
          name: p.prompt_name || p.name || "Untitled",
          description: p.description || "",
          template: p.prompt_template || p.template || "",
          version: p.version ?? 1,
          variables: extractVariables(p.prompt_template || p.template || ""),
          created_by: p.created_by || "—",
          created_at: p.created_at || "",
          updated_at: p.updated_at || p.created_at || "",
        }));
        setPrompts(promptList);
      }
    } catch (err) {
      console.error("Error fetching prompts:", err);
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchPrompts();
  }, [fetchPrompts]);

  const handleCreate = async (values: any) => {
    if (!accessToken) return;
    setCreateLoading(true);
    try {
      const url = proxyBaseUrl ? `${proxyBaseUrl}/prompts` : `/prompts`;
      const res = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt_name: values.name,
          prompt_template: values.template,
          description: values.description,
        }),
      });
      if (res.ok) {
        message.success("Prompt created successfully");
        setCreateModalOpen(false);
        form.resetFields();
        fetchPrompts();
      } else {
        const err = await res.json().catch(() => ({}));
        message.error(err.detail || "Failed to create prompt");
      }
    } catch {
      message.error("Network error creating prompt");
    } finally {
      setCreateLoading(false);
    }
  };

  const fetchVersions = async (promptName: string) => {
    if (!accessToken || versions[promptName]) return;
    setVersionsLoading((prev) => ({ ...prev, [promptName]: true }));
    try {
      const url = proxyBaseUrl
        ? `${proxyBaseUrl}/prompts/${encodeURIComponent(promptName)}/versions`
        : `/prompts/${encodeURIComponent(promptName)}/versions`;
      const res = await fetch(url, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });
      if (res.ok) {
        const data = await res.json();
        setVersions((prev) => ({
          ...prev,
          [promptName]: data.versions || data || [],
        }));
      }
    } catch {
      // Versions endpoint may not be available
    } finally {
      setVersionsLoading((prev) => ({ ...prev, [promptName]: false }));
    }
  };

  const handleRenderTest = async () => {
    if (!accessToken || !renderPrompt) return;
    setRenderLoading(true);
    try {
      const url = proxyBaseUrl
        ? `${proxyBaseUrl}/prompts/${encodeURIComponent(renderPrompt.name)}/render`
        : `/prompts/${encodeURIComponent(renderPrompt.name)}/render`;
      const res = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ variables: renderVariables }),
      });
      if (res.ok) {
        const data = await res.json();
        setRenderOutput(data.rendered || data.output || JSON.stringify(data));
      } else {
        // Fallback: client-side render
        let output = renderPrompt.template;
        for (const [key, val] of Object.entries(renderVariables)) {
          output = output.replace(new RegExp(`\\{\\{${key}\\}\\}`, "g"), val);
        }
        setRenderOutput(output);
      }
    } catch {
      // Client-side fallback
      let output = renderPrompt.template;
      for (const [key, val] of Object.entries(renderVariables)) {
        output = output.replace(new RegExp(`\\{\\{${key}\\}\\}`, "g"), val);
      }
      setRenderOutput(output);
    } finally {
      setRenderLoading(false);
    }
  };

  const handleRollback = async (promptName: string, version: number) => {
    if (!accessToken) return;
    try {
      const url = proxyBaseUrl
        ? `${proxyBaseUrl}/prompts/${encodeURIComponent(promptName)}/rollback`
        : `/prompts/${encodeURIComponent(promptName)}/rollback`;
      const res = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ version }),
      });
      if (res.ok) {
        message.success(`Rolled back to version ${version}`);
        // Refresh
        setVersions((prev) => {
          const copy = { ...prev };
          delete copy[promptName];
          return copy;
        });
        fetchPrompts();
      } else {
        message.error("Failed to rollback prompt");
      }
    } catch {
      message.error("Network error during rollback");
    }
  };

  const openRenderTest = (prompt: PromptRecord) => {
    setRenderPrompt(prompt);
    const vars: Record<string, string> = {};
    prompt.variables.forEach((v) => {
      vars[v] = "";
    });
    setRenderVariables(vars);
    setRenderOutput("");
    setRenderModalOpen(true);
  };

  const columns = [
    {
      title: "Name",
      dataIndex: "name",
      key: "name",
      render: (text: string) => <span className="font-medium">{text}</span>,
    },
    {
      title: "Description",
      dataIndex: "description",
      key: "description",
      ellipsis: true,
    },
    {
      title: "Version",
      dataIndex: "version",
      key: "version",
      width: 80,
      render: (v: number) => <Tag color="geekblue">v{v}</Tag>,
    },
    {
      title: "Variables",
      dataIndex: "variables",
      key: "variables",
      render: (vars: string[]) =>
        vars.length > 0
          ? vars.map((v) => (
              <Tag key={v} color="cyan">
                {`{{${v}}}`}
              </Tag>
            ))
          : "—",
    },
    {
      title: "Updated",
      dataIndex: "updated_at",
      key: "updated_at",
      render: (val: string) =>
        val ? new Date(val).toLocaleDateString() : "—",
    },
    {
      title: "Created By",
      dataIndex: "created_by",
      key: "created_by",
      ellipsis: true,
    },
    {
      title: "Actions",
      key: "actions",
      width: 180,
      render: (_: any, record: PromptRecord) => (
        <div className="flex gap-2">
          <Button
            size="small"
            icon={<PlayCircleOutlined />}
            onClick={() => openRenderTest(record)}
          >
            Test
          </Button>
          <Button
            size="small"
            icon={<HistoryOutlined />}
            onClick={() => fetchVersions(record.name)}
          >
            History
          </Button>
        </div>
      ),
    },
  ];

  const expandedRowRender = (record: PromptRecord) => {
    const promptVersions = versions[record.name];
    const isLoadingVersions = versionsLoading[record.name];

    return (
      <div className="p-4 space-y-4">
        <Descriptions column={1} size="small" bordered>
          <Descriptions.Item label="Template">
            <pre className="font-mono text-sm whitespace-pre-wrap bg-gray-50 p-3 rounded m-0">
              {record.template || "No template defined"}
            </pre>
          </Descriptions.Item>
        </Descriptions>

        {isLoadingVersions && <Spin size="small" />}

        {promptVersions && promptVersions.length > 0 && (
          <Card title="Version History" size="small">
            <Table
              dataSource={promptVersions}
              rowKey="version"
              size="small"
              pagination={false}
              columns={[
                {
                  title: "Version",
                  dataIndex: "version",
                  key: "version",
                  width: 80,
                  render: (v: number) => <Tag color="geekblue">v{v}</Tag>,
                },
                {
                  title: "Template",
                  dataIndex: "template",
                  key: "template",
                  ellipsis: true,
                  render: (t: string) => (
                    <span className="font-mono text-xs">{t}</span>
                  ),
                },
                {
                  title: "Created",
                  dataIndex: "created_at",
                  key: "created_at",
                  width: 140,
                  render: (val: string) =>
                    val ? new Date(val).toLocaleDateString() : "—",
                },
                {
                  title: "Actions",
                  key: "actions",
                  width: 100,
                  render: (_: any, ver: VersionRecord) =>
                    ver.version !== record.version ? (
                      <Popconfirm
                        title={`Rollback to version ${ver.version}?`}
                        onConfirm={() =>
                          handleRollback(record.name, ver.version)
                        }
                      >
                        <Button
                          size="small"
                          icon={<RollbackOutlined />}
                          danger
                        >
                          Rollback
                        </Button>
                      </Popconfirm>
                    ) : (
                      <Tag color="green">Current</Tag>
                    ),
                },
              ]}
            />
          </Card>
        )}

        {promptVersions && promptVersions.length === 0 && (
          <Empty description="No version history available" />
        )}
      </div>
    );
  };

  return (
    <div className="w-full p-6">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-semibold flex items-center gap-2">
          <FileTextOutlined /> Prompt Management
        </h1>
        <Button
          type="primary"
          icon={<PlusOutlined />}
          onClick={() => setCreateModalOpen(true)}
        >
          Create Prompt
        </Button>
      </div>

      {/* Prompts Table */}
      <Card>
        <Table
          dataSource={prompts}
          columns={columns}
          rowKey="prompt_id"
          loading={loading}
          pagination={{ pageSize: 10 }}
          size="small"
          expandable={{
            expandedRowRender,
            rowExpandable: () => true,
          }}
          locale={{ emptyText: "No prompts yet. Click 'Create Prompt' to get started." }}
        />
      </Card>

      {/* Create Prompt Modal */}
      <Modal
        title="Create New Prompt"
        open={createModalOpen}
        onCancel={() => {
          setCreateModalOpen(false);
          form.resetFields();
        }}
        footer={null}
        destroyOnClose
        width={640}
      >
        <Form form={form} layout="vertical" onFinish={handleCreate}>
          <Form.Item
            name="name"
            label="Prompt Name"
            rules={[{ required: true, message: "Please enter prompt name" }]}
          >
            <Input placeholder="e.g., summarize-email" />
          </Form.Item>
          <Form.Item name="description" label="Description">
            <Input placeholder="What does this prompt do?" />
          </Form.Item>
          <Form.Item
            name="template"
            label="Template"
            rules={[{ required: true, message: "Please enter template" }]}
            extra="Use {{variable_name}} for template variables"
          >
            <Input.TextArea
              rows={8}
              placeholder={"Summarize the following text:\n\n{{text}}\n\nProvide a {{length}} summary."}
              style={{ fontFamily: "monospace", fontSize: 13 }}
            />
          </Form.Item>
          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={createLoading}
              block
            >
              Create Prompt
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      {/* Render Test Modal */}
      <Modal
        title={`Test: ${renderPrompt?.name || ""}`}
        open={renderModalOpen}
        onCancel={() => setRenderModalOpen(false)}
        footer={null}
        destroyOnClose
        width={640}
      >
        {renderPrompt && (
          <div className="space-y-4">
            <Card size="small" title="Template">
              <pre className="font-mono text-sm whitespace-pre-wrap bg-gray-50 p-3 rounded m-0">
                {renderPrompt.template}
              </pre>
            </Card>

            {renderPrompt.variables.length > 0 ? (
              <>
                <Card size="small" title="Variables">
                  {renderPrompt.variables.map((v) => (
                    <div key={v} className="mb-2">
                      <label className="text-sm font-medium block mb-1">
                        {`{{${v}}}`}
                      </label>
                      <Input
                        value={renderVariables[v] || ""}
                        onChange={(e) =>
                          setRenderVariables((prev) => ({
                            ...prev,
                            [v]: e.target.value,
                          }))
                        }
                        placeholder={`Enter value for ${v}`}
                      />
                    </div>
                  ))}
                </Card>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={handleRenderTest}
                  loading={renderLoading}
                  block
                >
                  Render
                </Button>
              </>
            ) : (
              <Empty description="No variables detected in template" />
            )}

            {renderOutput && (
              <Card size="small" title="Rendered Output">
                <pre className="font-mono text-sm whitespace-pre-wrap bg-green-50 p-3 rounded m-0">
                  {renderOutput}
                </pre>
              </Card>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default PromptManagementDashboard;
