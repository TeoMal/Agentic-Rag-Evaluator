// Hackathon 2 -- Azure infrastructure (course units 35-38, extended).
//
// Deployed TWICE per release by scripts/deploy.py (locally and from cd.yml):
//   deployApp=false  registry, Log Analytics, Application Insights, Container Apps
//                    environment -- so the registry exists before the image is pushed
//   deployApp=true   the same, plus the Container App on the new image tag
// ARM's default Incremental mode leaves the Container App untouched on the first pass.

@description('Region for every resource.')
param location string = resourceGroup().location

@description('Container App name; also used to name the other resources.')
@minLength(2)
@maxLength(32)
param appName string = 'hackathon2-app'

@description('Globally unique registry name, 5-50 lowercase alphanumerics. Default is derived from the resource group.')
@minLength(5)
@maxLength(50)
param acrName string = 'acr${uniqueString(resourceGroup().id)}'

param imageName string = 'hackathon2-app'
param imageTag string = 'dev'

@description('false = infrastructure only (first pass); true = also deploy the Container App.')
param deployApp bool = false

param azureOpenAiEndpoint string = ''
param azureOpenAiApiVersion string = ''
param azureOpenAiDeployment string = ''
param azureOpenAiEmbeddingDeployment string = ''
@secure()
param azureOpenAiApiKey string = ''

// Pinned to exactly one replica: without Postgres the agent keeps HITL checkpoints
// and the vector index in memory, which scale-to-zero or a second replica would split or lose.
@minValue(0)
param minReplicas int = 1
@minValue(1)
param maxReplicas int = 1

@description('Marks everything this template owns: `deploy.py teardown` deletes exactly these, so a shared resource group (course subscription) survives.')
param tags object = {
  project: 'hackathon2'
}

// 1. Container registry (admin credentials, as in the course: works with a
//    Contributor-only service principal, which cannot create role assignments).
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

// 2. Log Analytics -- container logs and the store behind Application Insights.
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'law-${appName}'
  location: location
  tags: tags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// 3. Application Insights -- traces, requests, exceptions (handout section 11).
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${appName}'
  location: location
  tags: tags
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
  }
}

// 4. Container Apps environment, shipping stdout/stderr to Log Analytics.
resource containerEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${appName}'
  location: location
  tags: tags
  properties: {
    zoneRedundant: false
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// 5. The Container App -- second pass only.
resource containerApp 'Microsoft.App/containerApps@2024-03-01' = if (deployApp) {
  name: appName
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
      }
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'container-registry-password'
        }
      ]
      secrets: [
        {
          name: 'container-registry-password'
          value: acr.listCredentials().passwords[0].value
        }
        {
          name: 'azure-openai-api-key'
          value: azureOpenAiApiKey
        }
        {
          name: 'appinsights-connection-string'
          value: appInsights.properties.ConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'app'
          image: '${acr.properties.loginServer}/${imageName}:${imageTag}'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'APP_ENV'
              value: 'production'
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: azureOpenAiEndpoint
            }
            {
              name: 'OPENAI_API_VERSION'
              value: azureOpenAiApiVersion
            }
            {
              name: 'AZURE_OPENAI_DEPLOYMENT_NAME'
              value: azureOpenAiDeployment
            }
            {
              name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT'
              value: azureOpenAiEmbeddingDeployment
            }
            {
              name: 'AZURE_OPENAI_API_KEY'
              secretRef: 'azure-openai-api-key'
            }
            {
              name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
              secretRef: 'appinsights-connection-string'
            }
          ]
          probes: [
            {
              // Up to 2 minutes to boot (model clients, index warm-up) before liveness applies.
              type: 'Startup'
              httpGet: {
                path: '/'
                port: 8000
              }
              periodSeconds: 5
              failureThreshold: 24
            }
            {
              type: 'Liveness'
              httpGet: {
                path: '/'
                port: 8000
              }
              periodSeconds: 30
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/'
                port: 8000
              }
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
      }
    }
  }
}

output acrName string = acr.name
output acrLoginServer string = acr.properties.loginServer
output appInsightsName string = appInsights.name
output appFqdn string = deployApp ? containerApp!.properties.configuration.ingress.fqdn : ''
