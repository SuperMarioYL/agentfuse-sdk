from agentfuse.budget import Budget
from agentfuse.exceptions import BudgetExceeded
budget=Budget(ceiling_usd=1.0,name='fixture')
reservation=budget.reserve(0.6,estimated_tokens=100)
print(f'pending before completion: {budget.snapshot().pending_usd:.2f}')
try:
 budget.reserve(0.5)
except BudgetExceeded:print('second reservation: blocked before delegation')
budget.commit(0.4,actual_tokens=80,reservation=reservation)
s=budget.snapshot();print(f'spent: {s.spent_usd:.2f}; pending: {s.pending_usd:.2f}; remaining: {s.remaining_usd:.2f}')
