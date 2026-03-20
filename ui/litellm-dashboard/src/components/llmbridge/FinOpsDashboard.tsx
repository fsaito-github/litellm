"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Card,
  Typography,
  Select,
  Statistic,
  Row,
  Col,
  Spin,
  message,
} from "antd";
import {
  DollarOutlined,
  RiseOutlined,
  FallOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import { BarChart, AreaChart, DonutChart } from "@tremor/react";
import {
  proxyBaseUrl,
} from "../networking";

const { Title, Text } = Typography;

const globalLitellmHeaderName = "Authorization";

interface FinOpsDashboardProps {
  accessToken: string | null;
}

interface SpendSummary {
  total_spend: number;
  daily_average: number;
  by_model: { model: string; spend: number }[];
  by_team: { team: string; spend: number }[];
  by_organization: { organization: string; spend: number }[];
}

interface SpendTrend {
  date: string;
  spend: number;
}

interface Forecast {
  projected_monthly: number;
  projected_daily: number;
  trend_direction: "up" | "down" | "stable";
  trend_percentage: number;
}

const currencyFormatter = (value: number) => `$${value.toFixed(2)}`;

const FinOpsDashboard: React.FC<FinOpsDashboardProps> = ({ accessToken }) => {
  const [period, setPeriod] = useState("30d");
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<SpendSummary | null>(null);
  const [trends, setTrends] = useState<SpendTrend[]>([]);
  const [forecast, setForecast] = useState<Forecast | null>(null);

  const fetchData = useCallback(async () => {
    if (!accessToken) return;
    setLoading(true);

    const headers: Record<string, string> = {
      [globalLitellmHeaderName]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    };

    const base = proxyBaseUrl ?? "";

    try {
      const [summaryRes, trendsRes, forecastRes] = await Promise.all([
        fetch(`${base}/finops/summary?period=${period}`, {
          method: "GET",
          headers,
        }),
        fetch(`${base}/finops/trends?period=${period}`, {
          method: "GET",
          headers,
        }),
        fetch(`${base}/finops/forecast?period=${period}`, {
          method: "GET",
          headers,
        }),
      ]);

      if (!summaryRes.ok || !trendsRes.ok || !forecastRes.ok) {
        throw new Error("Failed to fetch FinOps data");
      }

      const [summaryData, trendsData, forecastData] = await Promise.all([
        summaryRes.json(),
        trendsRes.json(),
        forecastRes.json(),
      ]);

      setSummary(summaryData);
      setTrends(trendsData.trends ?? trendsData);
      setForecast(forecastData);
    } catch (err: any) {
      console.error("FinOps fetch error:", err);
      message.error(err.message ?? "Failed to load FinOps data");
    } finally {
      setLoading(false);
    }
  }, [accessToken, period]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const trendIcon =
    forecast?.trend_direction === "up" ? (
      <RiseOutlined style={{ color: "#cf1322" }} />
    ) : forecast?.trend_direction === "down" ? (
      <FallOutlined style={{ color: "#3f8600" }} />
    ) : null;

  if (loading) {
    return (
      <div className="flex items-center justify-center" style={{ minHeight: 400 }}>
        <Spin size="large" tip="Loading FinOps data..." />
      </div>
    );
  }

  return (
    <div className="p-4 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <Title level={3} style={{ margin: 0 }}>
          <DollarOutlined className="mr-2" />
          FinOps Dashboard
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

      {/* KPI Cards */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Total Spend"
              value={summary?.total_spend ?? 0}
              precision={2}
              prefix="$"
              valueStyle={{ color: "#1677ff" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Projected Monthly"
              value={forecast?.projected_monthly ?? 0}
              precision={2}
              prefix="$"
              valueStyle={{ color: "#722ed1" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Daily Average"
              value={summary?.daily_average ?? 0}
              precision={2}
              prefix="$"
              valueStyle={{ color: "#13c2c2" }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="Trend"
              value={forecast?.trend_percentage ?? 0}
              precision={1}
              suffix="%"
              prefix={trendIcon}
              valueStyle={{
                color:
                  forecast?.trend_direction === "up"
                    ? "#cf1322"
                    : forecast?.trend_direction === "down"
                      ? "#3f8600"
                      : "#8c8c8c",
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* Spend Over Time */}
      <Card className="mb-6" title="Spend Over Time">
        <AreaChart
          data={trends}
          index="date"
          categories={["spend"]}
          colors={["blue"]}
          valueFormatter={currencyFormatter}
          yAxisWidth={80}
          className="h-72"
          showAnimation
        />
      </Card>

      {/* Spend by Model + Spend by Team */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
        <Card title="Spend by Model">
          <DonutChart
            data={summary?.by_model ?? []}
            category="spend"
            index="model"
            valueFormatter={currencyFormatter}
            className="h-64"
            showAnimation
          />
        </Card>

        <Card title="Spend by Team">
          <BarChart
            data={summary?.by_team ?? []}
            index="team"
            categories={["spend"]}
            colors={["indigo"]}
            valueFormatter={currencyFormatter}
            yAxisWidth={80}
            className="h-64"
            showAnimation
          />
        </Card>
      </div>

      {/* Spend by Organization */}
      {summary?.by_organization && summary.by_organization.length > 0 && (
        <Card className="mb-6" title="Spend by Organization">
          <BarChart
            data={summary.by_organization}
            index="organization"
            categories={["spend"]}
            colors={["emerald"]}
            valueFormatter={currencyFormatter}
            yAxisWidth={80}
            className="h-64"
            showAnimation
          />
        </Card>
      )}

      {/* Forecast Details */}
      {forecast && (
        <Card title="Forecast">
          <Row gutter={[16, 16]}>
            <Col xs={24} sm={8}>
              <Statistic
                title="Projected Monthly Spend"
                value={forecast.projected_monthly}
                precision={2}
                prefix="$"
              />
            </Col>
            <Col xs={24} sm={8}>
              <Statistic
                title="Projected Daily Spend"
                value={forecast.projected_daily}
                precision={2}
                prefix="$"
              />
            </Col>
            <Col xs={24} sm={8}>
              <Statistic
                title="Trend Direction"
                value={forecast.trend_direction}
                prefix={trendIcon}
                valueStyle={{
                  textTransform: "capitalize",
                  color:
                    forecast.trend_direction === "up"
                      ? "#cf1322"
                      : forecast.trend_direction === "down"
                        ? "#3f8600"
                        : "#8c8c8c",
                }}
              />
            </Col>
          </Row>
        </Card>
      )}
    </div>
  );
};

export default FinOpsDashboard;
