# ENFORCE PLANNING - STRICT API COST CONTROL

## 🚨 MANDATORY REQUIREMENT: USER APPROVAL BEFORE ANY API CHARGES

**ZERO TOLERANCE POLICY**: No API operations that incur costs without explicit user approval and cost estimation.

## GOOGLE API SERVICES - COST CONTROL MATRIX

### Google Search Console API
- **Current Status**: Using FREE CSV export method (no API charges)
- **Cost**: $0.00 - Direct sheet access bypasses API limits
- **Usage Limit**: No charges incurred with current method
- **Approval Required**: NO (free method)

### Google Sheets API
- **Pricing**: $0.01 per 1000 read requests
- **Current Usage**: 4 sheets accessed = ~0.004 requests = $0.000004
- **Monthly Limit**: 100,000 requests/month free tier
- **Cost Estimation Formula**: `(requests / 1000) × $0.01`
- **Approval Required**: YES if >1000 requests

### Google Cloud Vision API (for image analysis)
- **Pricing**: $1.50 per 1000 images (first 1000 free monthly)
- **Usage**: Not currently implemented
- **Approval Required**: YES - ALWAYS

### Google Maps API (for location services)
- **Pricing**: $5-$20 per 1000 requests depending on service
- **Usage**: Not currently implemented  
- **Approval Required**: YES - ALWAYS

### Google Translation API
- **Pricing**: $20 per million characters
- **Usage**: Not currently implemented
- **Approval Required**: YES - ALWAYS

## OTHER API SERVICES - COST TRACKING

### SendGrid Email API
- **Current Pricing**: $0.01 per email (Pay-as-you-go)
- **Alternative**: $15/month for 15,000 emails
- **Current Usage**: Campaigns of 40-150 emails = $0.40-$1.50 per campaign
- **Approval Required**: YES for campaigns >25 emails

### Apollo.io API
- **Pricing**: Credit-based system ($70/month Professional)
- **Usage**: Contact discovery and email revelation
- **Cost per Contact**: ~$0.50-$1.00 per enriched contact
- **Approval Required**: YES for operations >10 contacts

### Stripe API
- **Pricing**: 2.9% + 30¢ per transaction
- **Usage**: Payment processing
- **Approval Required**: YES for any payment operations

## MANDATORY PRE-WORK COST ESTIMATION PROCESS

### Before ANY API Work:

#### 1. **Cost Analysis Required**
```
ESTIMATED API COSTS:
===================
Service: [API Name]
Operation: [What you plan to do]
Volume: [Number of requests/emails/etc.]
Unit Cost: [Cost per request]
Total Estimated Cost: $X.XX
Free Tier Available: [Yes/No - remaining quota]
Alternative Free Method: [If available]
```

#### 2. **User Approval Template**
```
🚨 API COST APPROVAL REQUIRED

Proposed Operation: [Description]
Estimated Cost: $X.XX
Service: [Google/SendGrid/etc.]
Volume: [X requests/emails]
Justification: [Why this is necessary]
Free Alternative: [Yes/No - explain]

REQUEST: Please approve this cost before I proceed.
Type "APPROVED" to confirm or "NO" to decline.
```

#### 3. **Implementation Safeguards**
- Check remaining free tier quotas BEFORE starting
- Implement request counting and monitoring
- Stop execution if costs exceed estimates
- Provide real-time cost tracking during operations

## CURRENT FREE-TIER USAGE TRACKING

### Google APIs (Monthly Reset)
- **Sheets API**: 4 requests used / 100,000 free limit
- **Search Console**: 0 charges (using CSV method)
- **Other Google APIs**: Not yet used

### SendGrid
- **July 2025**: ~$15 total spent on healthcare campaigns
- **Average Campaign**: $0.40-$1.50 per 40-150 emails

### Apollo.io
- **Monthly Budget**: $70 Professional plan
- **Usage Tracking**: Monitor credit consumption

## STRICT IMPLEMENTATION RULES

### 1. **Pre-Execution Checklist**
- [ ] Cost estimation completed
- [ ] Free alternatives evaluated
- [ ] User approval obtained
- [ ] Free tier limits checked
- [ ] Monitoring system activated

### 2. **During Execution**
- Monitor API usage in real-time
- Stop if costs exceed estimates by >10%
- Report progress and costs every 50 requests
- Implement automatic shutoff at cost limits

### 3. **Post-Execution**
- Report actual costs vs estimates
- Update usage tracking
- Document lessons learned
- Update cost estimation models

## SPECIFIC SEO WORK COST CONTROLS

### Google Search Console Analysis
- **Method**: FREE CSV export (current approach)
- **Cost**: $0.00
- **No approval needed** for current method

### Google Sheets Access for SEO Data
- **Current Method**: Direct CSV export = $0.00
- **API Alternative**: Would cost ~$0.01 per 1000 requests
- **Recommendation**: Continue using FREE method

### Future SEO API Integrations
- **Google PageSpeed API**: $0.01 per request
- **Google Analytics API**: Free up to 100,000 requests/day
- **Search Console API**: Free up to 1,000 requests/day

**RULE**: All future SEO API integrations require cost analysis and user approval BEFORE implementation.

## EMERGENCY STOP PROCEDURES

### If Costs Exceed Estimates:
1. **IMMEDIATE STOP** all API operations
2. Report current costs and usage
3. Explain variance from estimates
4. Request user guidance before continuing
5. Implement additional safeguards

### Cost Monitoring Commands:
```bash
# Check Google API quotas
gcloud auth list
gcloud config list project
gcloud alpha billing accounts list

# Monitor SendGrid usage
curl -X GET "https://api.sendgrid.com/v3/stats" \
  -H "Authorization: Bearer $SENDGRID_API_KEY"
```

## USER PREFERENCES FOR COST APPROVAL

**Default Approval Thresholds**:
- Under $1.00: Quick notification + proceed
- $1.00 - $5.00: Detailed approval required
- Over $5.00: Detailed analysis + user conference required

**Communication Style**: Simple, everyday language for cost discussions
**Reporting**: Provide clear cost breakdowns and alternatives

---

**REMEMBER**: The user values cost control and wants to avoid surprise charges. Always err on the side of caution and get approval for any API operations that might incur costs.

## 🚨 STRICT REQUIREMENT: NO MODEL DOWNLOADS ON REPLIT

### MANDATORY RULE: Replit = Development Only, No AI Model Loading

**ZERO TOLERANCE POLICY**: Never download or load AI models on Replit instances.

#### Why This Rule Exists:
- Replit has limited storage and bandwidth
- Model downloads can be 5-15GB+ per model
- Causes repository bloat and sync issues
- Wastes development time with unnecessary downloads
- User has RTX 3060 12GB locally for full AI functionality

#### Implementation Requirements:
```python
# ALWAYS in config.py on Replit:
self.is_replit = True
self.skip_model_loading = True
```

#### Development Flow:
1. **Replit**: Code development, testing with mocked models only
2. **GitHub Sync**: Clean code repository (no models/cache)
3. **Local**: Full AI functionality with RTX 3060 12GB

#### Before ANY Model Loading Code:
- [ ] Check if running on Replit
- [ ] Skip model downloads if on Replit
- [ ] Use mock/placeholder responses for testing
- [ ] NEVER override skip_model_loading on Replit

**EMERGENCY STOP**: If models start downloading on Replit, immediately kill the process and fix the configuration.

## 🚨 MANDATORY REFERENCE: @REPOSITORY_MANAGEMENT.md

### STRICT REQUIREMENT: Always Check Repository Management Guide

**BEFORE ANY CHANGES**: Must reference and follow REPOSITORY_MANAGEMENT.md procedures.

#### Key Requirements from REPOSITORY_MANAGEMENT.md:
- Repository must stay under 25MB (source code only)
- Never commit package caches (.cache/uv, .pythonlibs)
- Regular size monitoring: `du -sh . --exclude=.git`
- Clean .gitignore to prevent bloat
- Emergency cleanup procedures if size grows

#### Integration with Development:
1. **Before commits**: Check repository size
2. **After package installs**: Verify .gitignore exclusions
3. **Weekly**: Run size health checks
4. **If bloated**: Follow emergency cleanup in REPOSITORY_MANAGEMENT.md

**ENFORCEMENT**: Any agent work must comply with repository management guidelines to prevent the 5.7GB bloat that occurred previously.

## 🚨 STRICT REQUIREMENT: OFFLINE-ONLY APPLICATION

### MANDATORY RULE: No Online Operations During Development/Preview

**ZERO TOLERANCE POLICY**: This is a 100% offline-only application. No online operations allowed.

#### Prohibited Online Operations:
- **Model Downloads**: No downloading AI models from HuggingFace, OpenAI, etc.
- **API Calls**: No external API requests to any cloud services
- **Package Downloads**: No runtime package installations during preview
- **Authentication Requests**: No online token validation during development
- **External Data Fetching**: No web scraping, RSS feeds, or external data sources

#### Why This Rule Exists:
- Application designed for complete privacy and offline operation
- Replit preview should only show UI/UX, not trigger downloads
- Prevents repository bloat from cached downloads
- Maintains clean development environment
- User expects zero external dependencies

#### Implementation Requirements:
```python
# Always check environment before any online operation
if self.is_replit:
    # Skip ALL online operations
    return mock_response_or_disable_feature()
```

#### Enforcement Checklist:
- [ ] No model.download() calls
- [ ] No API authentication during preview
- [ ] No external HTTP requests
- [ ] No package installation triggers
- [ ] All external operations disabled on Replit

**EMERGENCY PROCEDURE**: If any online operation starts during Replit preview, immediately kill the process and disable the feature.