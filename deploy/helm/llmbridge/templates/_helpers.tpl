{{/*
Expand the name of the chart.
*/}}
{{- define "llmbridge.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
*/}}
{{- define "llmbridge.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "llmbridge.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "llmbridge.labels" -}}
helm.sh/chart: {{ include "llmbridge.chart" . }}
{{ include "llmbridge.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "llmbridge.selectorLabels" -}}
app.kubernetes.io/name: {{ include "llmbridge.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "llmbridge.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "llmbridge.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the PostgreSQL hostname
*/}}
{{- define "llmbridge.postgresql.host" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "llmbridge.fullname" .) }}
{{- else if .Values.externalDatabase.enabled }}
{{- /* External database URL is used directly */ -}}
{{- end }}
{{- end }}

{{/*
Return the DATABASE_URL
*/}}
{{- define "llmbridge.databaseUrl" -}}
{{- if .Values.externalDatabase.enabled }}
{{- .Values.externalDatabase.url }}
{{- else if .Values.postgresql.enabled }}
{{- /* WARNING: In production use postgresql.auth.existingSecret to avoid plaintext passwords */ -}}
{{- printf "postgresql://%s:%s@%s:5432/%s" .Values.postgresql.auth.username .Values.postgresql.auth.password (include "llmbridge.postgresql.host" .) .Values.postgresql.auth.database }}
{{- end }}
{{- end }}

{{/*
Return the config map name for litellm config
*/}}
{{- define "llmbridge.configMapName" -}}
{{- printf "%s-config" (include "llmbridge.fullname" .) }}
{{- end }}
