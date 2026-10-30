/* #507: 成长任务支持手动领取。
 *
 * 批量运行默认仍自动领奖；取消「自动领取」后只推进进度，任务停在 completed，
 * 由任务表里的「领取」按钮按账号逐个结算，避免所有积分在同一分钟到期。
 *
 *   node tests/_test_manual_claim.js
 */
const {dashboardHtml} = require('./_dashboard_source.js');
const html = dashboardHtml();

let pass = 0, fail = 0;
function check(label, ok, detail) {
  if (ok) { pass++; console.log('  ok   ' + label); }
  else { fail++; console.log('  FAIL ' + label + (detail ? '  ' + detail : '')); }
}

check('the auto-claim checkbox exists and defaults on',
      /id="growthAutoClaim" checked/.test(html));
check('the choice is persisted',
      /localStorage\.setItem\('wb-growth-auto-claim'/.test(html));
check('runGrowthTasks sends the checkbox state',
      /payload\.claim = !autoClaim \|\| autoClaim\.checked;/.test(html));
check('completed tasks render a manual claim button',
      /data-code="' \+ esc\(t\.task_code\) \+ '" onclick="claimGrowthTask\(this\)"/.test(html));
check('claimGrowthTask posts to /tasks/claim',
      /postJSON\('\/tasks\/claim', \{uid, code\}\)/.test(html));

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const input = {checked: false, disabled: false};
const schedule = {textContent: ''};
const requests = [];
const errors = [];
let rejectSave = false;
const ctx = {
  document: {getElementById: id => ({growthAutoRun: input, growthAutoRunSchedule: schedule}[id] || null)},
  localStorage: {getItem: () => null},
  toast: (message, kind) => errors.push([message, kind]),
  getJSON: async () => ({growth_tasks_auto_run: true, growth_tasks_hours: [10, 20]}),
  postJSON: async (url, payload) => {
    requests.push([url, payload.growth_tasks_auto_run]);
    assert.equal(input.disabled, true, 'disable the switch while saving');
    if(rejectSave) throw new Error('offline');
    return {growth_tasks_auto_run: payload.growth_tasks_auto_run};
  }
};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(path.join(__dirname, '../assets/js/tasks.js'), 'utf8'), ctx);

(async () => {
  ctx.renderGrowthAutoRun({growth_tasks_auto_run: true, growth_tasks_hours: [9, 21]});
  assert.equal(input.checked, true);
  assert.equal(schedule.textContent, '09:00 / 21:00');
  ctx.renderGrowthAutoRun({});
  assert.equal(input.checked, false, 'missing settings must not enable automation');
  input.checked = true;
  await ctx.saveGrowthAutoRun(input);
  assert.equal(input.checked, true);
  assert.equal(input.disabled, false);
  input.checked = false;
  await ctx.saveGrowthAutoRun(input);
  assert.equal(input.checked, false, 'a stale scheduler response must not undo the save');
  assert.equal(input.disabled, false);
  rejectSave = true;
  input.checked = true;
  await ctx.saveGrowthAutoRun(input);
  assert.equal(input.checked, false, 'failed saves restore the previous value');
  assert.equal(input.disabled, false);
  assert.deepStrictEqual(requests, [['/settings/save', true], ['/settings/save', false], ['/settings/save', true]]);
  assert.equal(errors[0][1], 'bad');
  console.log('Auto-run switch: persistence, refresh and failure recovery passed');
  console.log('PASS=%d FAIL=%d', pass, fail);
  process.exitCode = fail ? 1 : 0;
})().catch(error => { console.error(error); process.exitCode = 1; });
